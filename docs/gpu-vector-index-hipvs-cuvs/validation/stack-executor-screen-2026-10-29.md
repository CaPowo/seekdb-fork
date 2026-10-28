# CAGRA query-path reuse: 256D screen (2026-10-29)

## Change under test

The CAGRA adaptor previously created a 32 MiB-stack pthread and joined it for
every search request. The adaptor now dispatches build/search/snapshot jobs to
one reusable large-stack worker. Its synchronous job handoff keeps caller-owned
query/result memory alive until completion. The single-query result vectors are
also retained per index. No bridge allocation strategy, SQL, or public FP32
semantics changed. The branch restores the frozen V5
`nn_descent_niter=100`; both A/B executables used the same original C bridge and
the same build/search settings.

## Screen setup and reproducible invocation

This is a diagnostic screen, not the contest evaluation: AMD Radeon Graphics
gfx1100 (ROCm 7.2.4; AMD EPYC 9334 host), 250,000 synthetic 256D vectors,
4,096 single-query SQL
statements for each of Top-10 and Top-100, with 100 synthetic queries repeated
to fill the 4,096-query workload. The query/recall script, generated workload,
and results were kept in `/workspace/amd-cagra-v5-validation/new-image/` on the
GPU instance. Each run used the same already-imported table and created a new
index. `niter=100`, graph degree 128, intermediate degree 256 were fixed for
both control and modified builds.

The control used the pre-change seekdb executable and the modified run used the
rebuilt executable. Both used the original bridge and the same runner:

```bash
bash /tmp/run_original_bridge_control.sh \
  prekernel_originalbridge_repeat100 NN_DESCENT 128 256 100
bash /tmp/run_original_bridge.sh \
  kernelreuse_originalbridge_repeat100 NN_DESCENT 128 256 100
```

Both runner invocations require the instance's existing dataset, built binary,
ROCm/hipVS installation, and MySQL client. Each emitted 8,193 `cuvs_serve`
trace records (the first query plus both 4,096-query sweeps), exactly
`4,096 * 2 + 1` GPU CAGRA serves.

## Results

| Path | Run | Build + first query (ms) | Top-10 QPS | Top-100 QPS | Recall@10 | Recall@100 | Duplicate-result queries |
|---|---:|---:|---:|---:|---:|---:|---:|
| Control | 1 | 59,118 | 108.73 | 111.06 | 0.942993 | 0.868333 | 0 / 0 |
| Control | 2 | 59,622 | 110.31 | 108.08 | 0.941992 | 0.866733 | 0 / 0 |
| Reused worker | 1 | 54,769 | 241.21 | 237.56 | 0.941992 | 0.866340 | 0 / 0 |
| Reused worker | 2 | 59,854 | 241.04 | 238.69 | 0.949048 | 0.866853 | 0 / 0 |

Two-run mean QPS was 109.52/109.57 (Top-10/Top-100) for control and
241.13/238.13 for the reused worker: **2.20x / 2.17x** per-query throughput
(2.19x geometric mean). Build + first-query time was 59.37s control and
57.31s modified, but two construction samples are not enough to claim a build
gain. Mean recall delta was +0.00303 at @10 and -0.00094 at @100; every sweep
returned all expected rows and had no duplicate IDs within a result row.

The absolute recall values are below some official dataset floors; this
synthetic 256D screen is not a substitute for SIFT250K/Hidden250K. No claim is
made here that official quality gates or the official build-speed target pass.
The result isolates query-path overhead and justifies retaining the worker
reuse for evaluation on the official datasets.

## Build-algorithm probes (not retained)

To check whether replacing NN-Descent could improve construction without more
parameter sweeps, the same runner also tested the other available CAGRA build
algorithms. These are exploratory runs, not official performance results:

| CAGRA build algorithm | Build + first query (ms) | Top-10 / Top-100 QPS | Recall@10 / @100 | Duplicate-result queries |
|---|---:|---:|---:|---:|
| `ITERATIVE_CAGRA_SEARCH` | 79,813 | 195.04 / 184.21 | 0.930005 / 0.863926 | 0 / 0 |
| `IVF_PQ` | 24,051 | 240.09 / 232.54 | 0.701978 / 0.647588 | 0 / 0 |

Both paths emitted 8,193 `cuvs_serve` records. `ITERATIVE_CAGRA_SEARCH` built
slower and reduced @10 recall; `IVF_PQ` was much faster but missed the recall
target by a wide margin. Neither was retained; production still uses the V5
NN-Descent build path.

## CPU-path regression check

On the same instance, a `lib=vsag` HNSW index was built over 2,000 rows from the
screen table and returned 10 approximate results. Trace inspection found no
`cuvs_serve` call for that query:

```text
PASS lib=vsag rows=2000 query_results=10 gpu_cuvs_route=0
```

## Rollback

Revert the commit containing this change to restore per-request pthread creation
and temporary per-query result vectors. No SQL interface, test input, scoring
logic, or hipVS/cuVS library was changed.
