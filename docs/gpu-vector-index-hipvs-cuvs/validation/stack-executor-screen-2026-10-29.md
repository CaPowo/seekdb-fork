# CAGRA query-path reuse: 256D screen (2026-10-29)

## Change under test

The CAGRA adaptor previously created a 32 MiB-stack pthread and joined it for
every search request. The adaptor now dispatches build/search/snapshot jobs to
one reusable large-stack worker. Its synchronous job handoff keeps caller-owned
query/result memory alive until completion. The single-query result vectors are
also retained per index. The bridge retains its CAGRA search parameters and
per-index device scratch buffers, protected by an index-local mutex; FP32 input,
results, SQL, and CAGRA search parameters are unchanged. The frozen V5 build
setting `nn_descent_niter=100` is used throughout these measurements.

## Screen setup and reproducible invocation

This is a diagnostic screen, not the contest evaluation: an AMD gfx1100 GPU
with ROCm 7.2.4, 250,000 synthetic 256D vectors, 4,096 single-query SQL
statements for each of Top-10 and Top-100, with 100 synthetic queries repeated
to fill the 4,096-query workload. The query/recall script, generated workload,
and results were kept in `/workspace/amd-cagra-v5-validation/new-image/` on the
GPU instance. Each run used the same already-imported table and created a new
index. `niter=100`, graph degree 128, intermediate degree 256 were fixed for
both control and modified builds.

On the instance, the recorded runner command was:

```bash
bash /workspace/amd-cagra-v5-validation/new-image/run_variant_256_nq4096.sh \
  kernelreuse_repeat100 NN_DESCENT 128 256 100
```

The control used the same runner and bridge source, with the pre-change
`seekdb` executable selected by `/tmp/run_pre_kernel.sh`:

```bash
bash /tmp/run_pre_kernel.sh prekernel_repeat100 NN_DESCENT 128 256 100
```

Both runner invocations require the instance's existing dataset, built binary,
ROCm/hipVS installation, and MySQL client. Each emitted 8,193 `cuvs_serve`
trace records (the first query plus both 4,096-query sweeps), exactly
`4,096 * 2 + 1` GPU CAGRA serves.

## Results

| Path | Run | Build + first query (ms) | Top-10 QPS | Top-100 QPS | Recall@10 | Recall@100 | Duplicate-result queries |
|---|---:|---:|---:|---:|---:|---:|---:|
| Control | 1 | 58,915 | 109.53 | 111.45 | 0.943018 | 0.867747 | 0 / 0 |
| Control | 2 | 59,545 | 111.10 | 109.39 | 0.944019 | 0.866450 | 0 / 0 |
| Reused worker | 1 | 59,042 | 249.30 | 246.63 | 0.941016 | 0.868223 | 0 / 0 |
| Reused worker | 2 | 59,884 | 238.57 | 234.97 | 0.945020 | 0.863433 | 0 / 0 |

Two-run mean QPS was 110.32/110.42 (Top-10/Top-100) for control and
243.94/240.80 for the reused worker: **2.21x / 2.18x** per-query throughput
(2.20x geometric mean). Build + first-query time was effectively unchanged:
59.23s control vs 59.46s modified. Mean recall delta was -0.00050 at @10 and
-0.00127 at @100; every sweep returned all expected rows and had no duplicate
IDs within a result row.

The absolute recall values are below some official dataset floors; this
synthetic 256D screen is not a substitute for SIFT250K/Hidden250K. No claim is
made here that official quality gates or the official build-speed target pass.
The result isolates query-path overhead and justifies retaining the worker
reuse for evaluation on the official datasets.

## CPU-path regression check

On the same instance, a `lib=vsag` HNSW index was built over 2,000 rows from the
screen table and returned 10 approximate results. Trace inspection found no
`cuvs_serve` call for that query:

```text
PASS lib=vsag rows=2000 query_results=10 gpu_cuvs_route=0
```

## Rollback

Revert the commit containing this change to restore per-request pthread creation
and per-call bridge scratch allocation. No SQL interface, test input, scoring
logic, or hipVS/cuVS library was changed.
