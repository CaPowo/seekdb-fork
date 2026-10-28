# CAGRA query-buffer reuse screen (2026-10-29)

## Hypothesis

`seekdb_cuvs_search()` previously allocated three RMM buffers and created/destroyed
`cuvsCagraSearchParams_t` on every single-query call. A local experiment cached
the query, ID, and distance buffers and search-parameter object in the bridge
handle, growing buffers only when a larger request arrived. A per-handle mutex
serialized access to this mutable scratch state. Search parameters, CAGRA
algorithm, SQL contract, and FP32 data were unchanged.

## Test

The bridge was compiled on AMD Radeon Graphics `gfx1100`, ROCm 7.2.4, with the
image-provided hipVS. Baseline bridge source was taken from commit
`4e4049799`; candidate source was local WIP commit `83857e1dc`. Both loaded the
same serialized FP32 CAGRA graph (250,000 x 256), built once with the unchanged
NN-Descent settings. Each process ran 32 warmups, then 4,096 single-query calls
for Top-10 and Top-100. Calls used the same deterministic query vectors.
Three runs per bridge alternated baseline/candidate order. The benchmark checked
all IDs were in range and each result row had no duplicates.

The exact timing summaries were:

| Bridge | Run | Top-10 elapsed ms | Top-10 QPS | Top-100 elapsed ms | Top-100 QPS | Duplicate rows |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 1 | 10,777.689 | 380.044 | 10,854.890 | 377.341 | 0 |
| Candidate | 1 | 10,591.704 | 386.718 | 10,797.214 | 379.357 | 0 |
| Candidate | 2 | 10,755.858 | 380.816 | 10,881.141 | 376.431 | 0 |
| Baseline | 2 | 10,647.747 | 384.682 | 10,861.525 | 377.111 | 0 |
| Candidate | 3 | 10,707.692 | 382.529 | 10,841.730 | 377.799 | 0 |
| Baseline | 3 | 10,702.122 | 382.728 | 10,898.629 | 375.827 | 0 |

Three-run means:

| Bridge | Top-10 QPS | Top-100 QPS | Geometric mean QPS |
|---|---:|---:|---:|
| Baseline | 382.485 | 376.760 | 379.611 |
| Candidate | 383.354 | 377.862 | 380.598 |

The candidate's measured gain was **0.23% Top-10**, **0.29% Top-100**, and
**0.26% geometric mean**. If it transferred unchanged to the official query
score, this would be roughly 0.013 of the 2 query points. This is below a
meaningful contest-score gain and within the noise expected from this short
screen, so the code was not promoted to `amd-cagra-submit`.

All six runs returned only in-range IDs and had zero duplicate-result rows.
Top-10 hashes were identical across runs. Top-100 hashes varied across runs for
both baseline and candidate, indicating nondeterministic approximate results;
no recall claim is made from this bridge-only test. A separate 10,000 x 32
serialization smoke test passed on both bridges with repeated Top-K sequence
10, 100, 10 on write and reload, including exact vector round-trip and top-1
ID 0.

## Reproduction

The reproducible harness is
[`benchmark_search.cpp`](../../../tools/hipvs_bridge/benchmark_search.cpp).
The six-run table was produced by its prototype; the committed harness was then
compiled on the GPU host and rerun against both libraries, producing 385.122 /
377.802 QPS for baseline and 388.804 / 381.022 QPS for candidate, with zero
duplicate rows. Bridge variants and the serialized graph were under
`/workspace/bridge-{baseline,candidate}` and `/workspace/cuvs-search-test/` on
the GPU instance. That fresh instance had no seekdb server build or prior 256D
SQL dataset, so this remains a bridge-level microbenchmark, not the contest SQL
benchmark or an official score measurement.

The core commands were:

```bash
g++ -std=c++17 -O2 -I tools/hipvs_bridge \
  tools/hipvs_bridge/benchmark_search.cpp \
  -L/workspace/bridge-baseline -lseekdb_cuvs_bridge \
  -Wl,-rpath,/workspace/bridge-baseline -o /workspace/cuvs-search-test/bench-baseline
LD_LIBRARY_PATH=/workspace/bridge-baseline:/opt/hipvs/lib:/opt/rocm/lib \
  /workspace/cuvs-search-test/bench-baseline build /workspace/cuvs-search-test/index.cuvs
LD_LIBRARY_PATH=/workspace/bridge-baseline:/opt/hipvs/lib:/opt/rocm/lib \
  /workspace/cuvs-search-test/bench-baseline bench /workspace/cuvs-search-test/index.cuvs
```

The candidate was compiled and run the same way, replacing `bridge-baseline`
with `bridge-candidate` and using the same `index.cuvs` file.

The candidate implementation remains available locally on branch
`experiment/cuvs-query-buffer-reuse` at commit `83857e1dc` for future profiling;
it is not part of the pushed contest branch.
