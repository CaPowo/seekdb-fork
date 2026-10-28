# AMD CAGRA screening checkpoint (2026-10-28)

## Scope and candidate

- V5 frozen base: `634888d9b037153fa143d2c913ba4b25503e1d62`.
- Initial candidate used NN-Descent `nn_descent_niter=50`; after repeat screening,
  current branch source uses `nn_descent_niter=55` (see the follow-up section below).
  All SQL/API data remains FP32.
- Local synthetic screening only: 250,000 vectors per dataset, dimensions 128 and 256, fixed generated query sets, top-10/top-100 ID recall against exact ground truth. This is not the organizer's formal dataset or score.

## Exact screening commands

On the cloud instance, with the V5 binary and generated 250K dataset already loaded:

```bash
/workspace/amd-cagra-v5-validation/new-image/run_variant_128.sh niter50b NN_DESCENT 128 256 50
/workspace/amd-cagra-v5-validation/new-image/run_variant_256.sh niter50 NN_DESCENT 128 256 50
```

Each runner creates a fresh vector table/index, times index creation plus the first query, runs the fixed 100-query top-10 and top-100 SQL batches, verifies the GPU `cuvs_serve` trace, checks result counts, and computes recall from `truth_top100.npy`. Both completed with exit status 0 in the captured runs.

Baseline configurations used the same synthetic input and query files with NN-Descent `niter=100`. The runs were screening runs, not a fully interleaved three-round evaluation.

## Literal measured output

128-D candidate (`niter50b`):

```text
niter50b,top10,elapsed_ms=722,qps=138.50
niter50b,top100,elapsed_ms=701,qps=142.65
niter50b,35797
baseline Recall@10=0.993000 duplicate_queries=0
baseline Recall@100=0.971800 duplicate_queries=0
candidate Recall@10=0.990000 duplicate_queries=0
candidate Recall@100=0.970600 duplicate_queries=0
128_niter50b delta@10=-0.003000
128_niter50b delta@100=-0.001200
```

Against the earlier 128-D niter-100 run (51,911 ms), the observed build-plus-first-query ratio is 1.45x.

256-D candidate (`niter50`, captured screening round):

```text
niter50,39297
baseline Recall@10=0.941000 duplicate_queries=0
baseline Recall@100=0.865500 duplicate_queries=0
candidate Recall@10=0.937000 duplicate_queries=0
candidate Recall@100=0.866300 duplicate_queries=0
niter50 delta@10=-0.004000
niter50 delta@100=+0.000800
```

Against the corresponding niter-100 256-D screening run (59,782 ms), the observed build-plus-first-query ratio is 1.52x. Both recall deltas remain within the 0.005 screening limit, but the top-10 margin is narrow.

## Broader algorithm experiments

| 256-D candidate | Build + first query | Recall@10 delta | Recall@100 delta | Result |
|---|---:|---:|---:|---|
| `AUTO_SELECT`, degree 128 / intermediate 256 | 29,259 ms | -0.0230 | -0.0187 | fails recall |
| `AUTO_SELECT`, degree 192 / intermediate 384 | 39,442 ms | +0.0170 | +0.0333 | recall passes; query throughput needs more screening |
| `AUTO_SELECT`, degree 160 / intermediate 320 | 20,364 ms graph snapshot | — | — | top-10 batch timed out; incomplete |
| `ITERATIVE_CAGRA_SEARCH` | 84,935 ms | 0.0000 | -0.0046 | slower than baseline |
| 50% seed + `cuvsCagraExtend` | 80,940 ms | -0.1410 | -0.1330 | fails recall and slower |

An FP16 bridge prototype compiled, but its first query did not complete in the screening timeout; it was discarded. None of these experimental settings are in the candidate source.

## 2026-10-28 follow-up: append-path trace and discarded experiment

Two source-level batching attempts targeted `ObVectorIndexSliceStore::append_row()` in
`src/storage/ddl/ob_direct_load_struct.cpp` and `ObHNSWIndexAppendBufferOperator` in
`src/storage/ddl/ob_ddl_pipeline.cpp`. Both compiled, but neither changed the exercised
hot path: three follow-up 256-D `niter=50` runs still recorded exactly 250,000
`add_dense ... b=1` trace events. The patches were archived and removed from the working
branch; they are not candidate optimizations.

Verified follow-up run (`core_batch_diag_n50_256`):

```text
core_batch_diag_n50_256,top10,elapsed_ms=924,qps=108.23
core_batch_diag_n50_256,top100,elapsed_ms=938,qps=106.61
core_batch_diag_n50_256,40849
baseline Recall@10=0.941000 duplicate_queries=0
baseline Recall@100=0.865500 duplicate_queries=0
candidate Recall@10=0.942000 duplicate_queries=0
candidate Recall@100=0.869000 duplicate_queries=0
core_batch_diag_n50_256 delta@10=+0.001000
core_batch_diag_n50_256 delta@100=+0.003500
```

The trace source is `src/oblib/lib/vector/ob_vsag_adaptor.cpp`: the actual CAGRA
ingest point is `ob_cuvs_add()`, which accumulates vectors in the per-index buffer;
the GPU graph is then built by `seekdb_cuvs_build()` on the first query. Follow-up
work should begin at this verified route, rather than the two unused DDL paths.

The additional trace/log archive is:

`/root/local-backups/amd-cagra-v5-validation/new-image/saved/amd-latest-pipeline-runs-20261028.tgz`

SHA-256: `183e637a6d315b3952aa4043e0addf9c411eaf926f278b15302e0a82a1380ef1`.
The unmerged patches are separately preserved as `direct-load-batch-attempt.patch`
and `hnsw-append-pipeline-attempt.patch` in the same local backup directory.

## 2026-10-28 follow-up: larger graph quality/query trade-off

The 256-D `NN_DESCENT`, degree 160 / intermediate 320, niter 50 candidate measured:

```text
nn160_320_n50_256,top10,elapsed_ms=1020,qps=98.04
nn160_320_n50_256,top100,elapsed_ms=1065,qps=93.90
nn160_320_n50_256,44401
baseline Recall@10=0.941000 duplicate_queries=0
baseline Recall@100=0.865500 duplicate_queries=0
candidate Recall@10=0.968000 duplicate_queries=0
candidate Recall@100=0.914300 duplicate_queries=0
nn160_320_n50_256 delta@10=+0.027000
nn160_320_n50_256 delta@100=+0.048800
```

Recall has ample headroom, but QPS is below 90% of the paired baseline examples
(119.62 / 112.49); this configuration is not accepted. Its archive is
`/root/local-backups/amd-cagra-v5-validation/new-image/saved/amd-nn160-320-result-20261028.tgz`
(SHA-256 `0f4525f91ee847d77a4adaa4e128d2f8ad5e280478b4afe647138bbb359c99b3`).

A follow-up with the same graph configuration and `search_width=3` measured 47,571 ms,
Recall@10/100 deltas +0.0250/+0.0499, and QPS 98.52/96.25. Query throughput still
misses the 90%-of-baseline floor, so reducing search width to 3 is also rejected.
The run is backed up at
`/root/local-backups/amd-cagra-v5-validation/new-image/saved/amd-nn160-320-sw3-result-20261028.tgz`
(SHA-256 `e25b15586810eb8f9bc1deb7f152daa2ceea56ad48d37dfd750346a0edf949c6`).

## 2026-10-28 follow-up: niter=55 retest and code-path experiments

Two 256-D NN-Descent niter=55 runs at the default 128/256 graph degrees passed the
local recall and query-throughput screens:

| Run | Build + first query | Recall@10 / @100 | Query QPS top10 / top100 |
|---|---:|---:|---:|
| `niter55` | 40,354 ms | 0.9420 / 0.8684 | 102.56 / 120.63 |
| `niter55_repeat_256` | 43,471 ms | 0.9450 / 0.8698 | 99.01 / 130.72 |

Both recall results exceed the frozen synthetic baseline values (0.9410 / 0.8655),
and the query QPS geometric means exceed 90% of the corresponding baseline example
(119.62 / 112.49). niter=55 is the current screening candidate in bridge source;
official hardware/data evaluation remains authoritative.

The repeated run outputs are backed up at
`/root/local-backups/amd-cagra-v5-validation/new-image/saved/amd-niter55-repeat-20261028.tgz`
(SHA-256 `fcf3d54eec7b1716faeeb514245a70423b197faf3c45e821cc370dba09b8ccdf`).

Other one-hour follow-ups were rejected:

- Registering the CPU input buffer with `cudaHostRegister` succeeded for 256,000,000
  bytes, but did not yield repeatable build or query gains. The experiment was
  reverted. Its archives are `amd-hostpin-n50-result-20261028.tgz` (SHA-256
  `d828f8bd2e0dd244df11f84fddea89fd6ee22557d1d4d7099e0b5a7404895df8`) and
  `amd-hostpin-diag-result-20261028.tgz` (SHA-256
  `b68c69bf8deb495783dbee66c56eefe5f5f39c427ff5f05dbc423549fd61171c`).
- Reducing graph degree to 112 / intermediate 224 at niter=50 built in 34,070 ms,
  but Recall@10/100 fell to 0.6010 / 0.5567; rejected. Archive:
  `amd-nn112-224-fail-20261028.tgz` (SHA-256
  `1a30b1c18d8813a674b5cfb6de4d6041a4b74150f122de7522be474496b7a0a7`).

## Artifact backup

The complete pulled cloud screening archive, including generated SQL, IDs, traces, and per-run logs, is:

`/root/local-backups/amd-cagra-v5-validation/new-image/saved/amd-cagra-v5-screen-results-20261028.tgz`

It was reopened with `tar -tzf`; SHA-256 is recorded alongside this report in the local backup directory. These exploratory runs do not substitute for the official fixed-dataset evaluation.
