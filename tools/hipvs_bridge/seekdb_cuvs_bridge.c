/* seekdb <-> hipVS bridge: plain-C CAGRA kNN; all cuVS/ROCm deps live here. */
#include "seekdb_cuvs_bridge.h"
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/cagra.h>
#include <dlpack/dlpack.h>
#ifdef __HIP_PLATFORM_AMD__
#include <cuvs/cuda_runtime.h>
#else
#include <cuda_runtime.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdint.h>

typedef struct {
  cuvsResources_t        res;
  cuvsCagraIndexParams_t ip;
  cuvsCagraIndex_t       idx;
  cuvsCagraSearchParams_t search_params;
  size_t                  default_itopk_size;
  float                  *query_device;
  unsigned               *ids_device;
  float                  *distances_device;
  size_t                  query_capacity;
  size_t                  ids_capacity;
  size_t                  distances_capacity;
  pthread_mutex_t         search_mutex;
  int                     search_mutex_initialized;
  long n, dim;
} sk_cuvs_index;

/* Reuse per-index scratch storage across repeated SQL queries. */
static int ensure_device_buffer(sk_cuvs_index* h, void** buffer,
                               size_t* capacity, size_t required)
{
  size_t next_capacity = *capacity ? *capacity : 4096;
  void* next_buffer = NULL;
  if (required <= *capacity) return 0;
  while (next_capacity < required) {
    if (next_capacity > SIZE_MAX / 2) {
      next_capacity = required;
      break;
    }
    next_capacity *= 2;
  }
  if (cuvsRMMAlloc(h->res, &next_buffer, next_capacity) != CUVS_SUCCESS)
    return -1;
  if (*buffer && cuvsRMMFree(h->res, *buffer, *capacity) != CUVS_SUCCESS) {
    cuvsRMMFree(h->res, next_buffer, next_capacity);
    return -1;
  }
  *buffer = next_buffer;
  *capacity = next_capacity;
  return 0;
}

void* seekdb_cuvs_build(const float* base, long n, long dim)
{
  sk_cuvs_index* h = (sk_cuvs_index*)calloc(1, sizeof(*h));
  if (!h) return NULL;
  if (pthread_mutex_init(&h->search_mutex, NULL) != 0) { free(h); return NULL; }
  h->search_mutex_initialized = 1;
  if (cuvsResourcesCreate(&h->res) != CUVS_SUCCESS) {
    pthread_mutex_destroy(&h->search_mutex);
    free(h);
    return NULL;
  }

  DLManagedTensor dset; long ds[2] = {n, dim};
  dset.dl_tensor.data = (void*)base;
  dset.dl_tensor.device.device_type = kDLCPU; dset.dl_tensor.device.device_id = 0;
  dset.dl_tensor.ndim = 2;
  dset.dl_tensor.dtype.code = kDLFloat; dset.dl_tensor.dtype.bits = 32; dset.dl_tensor.dtype.lanes = 1;
  dset.dl_tensor.shape = ds; dset.dl_tensor.strides = 0; dset.dl_tensor.byte_offset = 0;

  if (cuvsCagraIndexParamsCreate(&h->ip) != CUVS_SUCCESS) goto fail_res;
  h->ip->build_algo = NN_DESCENT;
  h->ip->graph_degree = 128;
  h->ip->intermediate_graph_degree = 256;
  h->ip->nn_descent_niter = 100;
  if (cuvsCagraIndexCreate(&h->idx)      != CUVS_SUCCESS) goto fail_ip;
  if (cuvsCagraBuild(h->res, h->ip, &dset, h->idx) != CUVS_SUCCESS) goto fail_idx;
  h->n = n; h->dim = dim;
  return h;

fail_idx: cuvsCagraIndexDestroy(h->idx);
fail_ip:  cuvsCagraIndexParamsDestroy(h->ip);
fail_res: cuvsResourcesDestroy(h->res);
  if (h->search_mutex_initialized) pthread_mutex_destroy(&h->search_mutex);
  free(h);
  fprintf(stderr, "[seekdb_cuvs] build failed: %s\n", cuvsGetLastErrorText());
  return NULL;
}

int seekdb_cuvs_search(void* handle, const float* query, long nq, long topk,
                       unsigned int* out_ids, float* out_dist)
{
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  if (!h || !query || !out_ids || nq <= 0 || topk <= 0 || h->dim <= 0 ||
      (size_t)nq > SIZE_MAX / (size_t)h->dim / sizeof(float) ||
      (size_t)nq > SIZE_MAX / (size_t)topk / sizeof(unsigned)) return 1;
  int rc = 0;
  if (pthread_mutex_lock(&h->search_mutex) != 0) return 2;
  if (ensure_device_buffer(h, (void**)&h->query_device, &h->query_capacity,
                           sizeof(float)*(size_t)nq*(size_t)h->dim) != 0) { rc=10; goto done; }
  if (ensure_device_buffer(h, (void**)&h->ids_device, &h->ids_capacity,
                           sizeof(unsigned)*(size_t)nq*(size_t)topk) != 0) { rc=11; goto done; }
  if (ensure_device_buffer(h, (void**)&h->distances_device,
                           &h->distances_capacity,
                           sizeof(float)*(size_t)nq*(size_t)topk) != 0) { rc=12; goto done; }
  if (cudaMemcpy(h->query_device, query, sizeof(float)*(size_t)nq*(size_t)h->dim,
                 cudaMemcpyDefault) != cudaSuccess) { rc=13; goto done; }
  {
    DLManagedTensor qt, nt, dt; long qs[2]={nq,h->dim}, nsh[2]={nq,topk}, dsh[2]={nq,topk};
    qt.dl_tensor.data=h->query_device; qt.dl_tensor.device.device_type=kDLCUDA; qt.dl_tensor.device.device_id=0;
    qt.dl_tensor.ndim=2; qt.dl_tensor.dtype.code=kDLFloat; qt.dl_tensor.dtype.bits=32; qt.dl_tensor.dtype.lanes=1;
    qt.dl_tensor.shape=qs; qt.dl_tensor.strides=0; qt.dl_tensor.byte_offset=0;
    nt=qt; nt.dl_tensor.data=h->ids_device; nt.dl_tensor.dtype.code=kDLUInt; nt.dl_tensor.shape=nsh;
    dt=qt; dt.dl_tensor.data=h->distances_device; dt.dl_tensor.dtype.code=kDLFloat; dt.dl_tensor.shape=dsh;
    if (!h->search_params) {
      if (cuvsCagraSearchParamsCreate(&h->search_params) != CUVS_SUCCESS) { rc=14; goto done; }
      h->default_itopk_size = h->search_params->itopk_size;
    }
    {
      size_t required_itopk = (size_t)topk * 4;
      if (required_itopk < 512) required_itopk = 512;
      h->search_params->itopk_size = h->default_itopk_size > required_itopk
                                          ? h->default_itopk_size : required_itopk;
      h->search_params->search_width = 4;
    }
    cuvsFilter f; f.type=NO_FILTER; f.addr=(uintptr_t)0;
    if (cuvsCagraSearch(h->res, h->search_params, h->idx, &qt, &nt, &dt, f) != CUVS_SUCCESS) { rc=15; goto done; }
    if (cudaMemcpy(out_ids, h->ids_device, sizeof(unsigned)*(size_t)nq*(size_t)topk, cudaMemcpyDefault) != cudaSuccess) { rc=16; goto done; }
    if (out_dist && cudaMemcpy(out_dist, h->distances_device, sizeof(float)*(size_t)nq*(size_t)topk, cudaMemcpyDefault) != cudaSuccess) { rc=17; goto done; }
  }
done:
  pthread_mutex_unlock(&h->search_mutex);
  if (rc) fprintf(stderr, "[seekdb_cuvs] search rc=%d: %s\n", rc, cuvsGetLastErrorText());
  return rc;
}

void seekdb_cuvs_free(void* handle)
{
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  if (!h) return;
  if (h->search_mutex_initialized) pthread_mutex_lock(&h->search_mutex);
  if (h->search_params) cuvsCagraSearchParamsDestroy(h->search_params);
  if (h->distances_device) cuvsRMMFree(h->res, h->distances_device, h->distances_capacity);
  if (h->ids_device) cuvsRMMFree(h->res, h->ids_device, h->ids_capacity);
  if (h->query_device) cuvsRMMFree(h->res, h->query_device, h->query_capacity);
  cuvsCagraIndexDestroy(h->idx);
  cuvsCagraIndexParamsDestroy(h->ip);
  cuvsResourcesDestroy(h->res);
  if (h->search_mutex_initialized) {
    pthread_mutex_unlock(&h->search_mutex);
    pthread_mutex_destroy(&h->search_mutex);
  }
  free(h);
}

int seekdb_cuvs_serialize(void* handle, const char* filename)
{
  sk_cuvs_index* index = (sk_cuvs_index*)handle;
  if (!index || !filename) return 1;
  if (pthread_mutex_lock(&index->search_mutex) != 0) return 2;
  if (cuvsCagraSerialize(index->res, filename, index->idx, true) != CUVS_SUCCESS) {
    fprintf(stderr, "[seekdb_cuvs] serialize failed: %s\n", cuvsGetLastErrorText());
    pthread_mutex_unlock(&index->search_mutex);
    return 1;
  }
  pthread_mutex_unlock(&index->search_mutex);
  return 0;
}

void* seekdb_cuvs_deserialize(const char* filename, long n, long dim)
{
  if (!filename || n <= 0 || dim <= 0) return NULL;
  sk_cuvs_index* index = (sk_cuvs_index*)calloc(1, sizeof(*index));
  if (!index) return NULL;
  if (pthread_mutex_init(&index->search_mutex, NULL) != 0) { free(index); return NULL; }
  index->search_mutex_initialized = 1;
  if (cuvsResourcesCreate(&index->res) != CUVS_SUCCESS) goto fail;
  if (cuvsCagraIndexParamsCreate(&index->ip) != CUVS_SUCCESS) goto fail_res;
  if (cuvsCagraIndexCreate(&index->idx) != CUVS_SUCCESS) goto fail_ip;
  if (cuvsCagraDeserialize(index->res, filename, index->idx) != CUVS_SUCCESS) goto fail_idx;
  {
    int64_t actual_n = 0;
    int64_t actual_dim = 0;
    if (cuvsCagraIndexGetSize(index->idx, &actual_n) != CUVS_SUCCESS ||
        cuvsCagraIndexGetDims(index->idx, &actual_dim) != CUVS_SUCCESS ||
        actual_n != n || actual_dim != dim) {
      fprintf(stderr, "[seekdb_cuvs] snapshot shape mismatch: %ld x %ld, expected %ld x %ld\n",
              (long)actual_n, (long)actual_dim, n, dim);
      goto fail_idx;
    }
  }
  index->n = n;
  index->dim = dim;
  return index;
fail_idx:
  cuvsCagraIndexDestroy(index->idx);
fail_ip:
  cuvsCagraIndexParamsDestroy(index->ip);
fail_res:
  cuvsResourcesDestroy(index->res);
fail:
  fprintf(stderr, "[seekdb_cuvs] deserialize failed: %s\n", cuvsGetLastErrorText());
  if (index->search_mutex_initialized) pthread_mutex_destroy(&index->search_mutex);
  free(index);
  return NULL;
}

int seekdb_cuvs_cagra_knn(const float* base, long n, long dim,
                          const float* query, long nq, long topk,
                          unsigned int* out_ids)
{
  void* h = seekdb_cuvs_build(base, n, dim);
  if (!h) return 10;
  int rc = seekdb_cuvs_search(h, query, nq, topk, out_ids, NULL);
  seekdb_cuvs_free(h);
  return rc;
}
