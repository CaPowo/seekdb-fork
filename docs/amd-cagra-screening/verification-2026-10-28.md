# AMD CAGRA screening checkpoint (2026-10-28)

## Scope and candidate

- V5 frozen base: `634888d9b037153fa143d2c913ba4b25503e1d62`.
- Candidate source: `tools/hipvs_bridge/seekdb_cuvs_bridge.c` changes NN-Descent `nn_descent_niter` from 100 to 50; all SQL/API data remains FP32.
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

## Artifact backup

The complete pulled cloud screening archive, including generated SQL, IDs, traces, and per-run logs, is:

`/root/local-backups/amd-cagra-v5-validation/new-image/saved/amd-cagra-v5-screen-results-20261028.tgz`

It was reopened with `tar -tzf`; SHA-256 is recorded alongside this report in the local backup directory. These exploratory runs do not substitute for the official fixed-dataset evaluation.
