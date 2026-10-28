# Host-registration build screen (2026-10-29)

## Hypothesis and prototype

The CAGRA bridge receives a pageable host FP32 vector buffer. This experiment
registered the 250,000 x 256 input buffer with `cudaHostRegister` for the
duration of `cuvsCagraBuild`, then unregistered it after the call. If hipVS
could use the registration for transfer, build time might fall without changing
the graph algorithm, settings, data, or SQL semantics. Registration succeeded
for the full 256,000,000-byte buffer on the test GPU.

The prototype was compiled separately from the contest branch. No
host-registration code was retained in `amd-cagra-submit`.

## Measurement

Environment: AMD Radeon Graphics `gfx1100`, ROCm 7.2.4, image-provided hipVS.
Baseline and prototype used the same deterministic FP32 250K x 256 vectors,
NN-Descent settings (`graph_degree=128`, `intermediate_graph_degree=256`,
`niter=100`), and bridge-level build path. Each build was followed by one
Top-10 search; all 10 IDs were in range and unique, with the same top-1 ID and
distance in all runs.

| Order | Variant | Build time (ms) | Host registration | Top-1 ID | ID integrity |
|---:|---|---:|---|---:|---|
| 1 | Baseline | 42,646.897 | n/a | 51,482 | PASS |
| 2 | Registered input | 42,933.134 | success, 256,000,000 bytes | 51,482 | PASS |
| 3 | Registered input | 41,641.660 | success, 256,000,000 bytes | 51,482 | PASS |
| 4 | Baseline | 41,969.825 | n/a | 51,482 | PASS |
| 5 | Baseline | 43,167.826 | n/a | 51,482 | PASS |
| 6 | Registered input | 43,643.091 | success, 256,000,000 bytes | 51,482 | PASS |

Three-run means were 42,594.849 ms for baseline and 42,739.295 ms for the
registered-input prototype: **0.34% slower**. Run-to-run spread is larger than
that difference, and the first registered run was slower. This does not
support a repeatable build improvement, so host registration was not promoted.

The bridge-level harness is
[`benchmark_build.cpp`](../../../tools/hipvs_bridge/benchmark_build.cpp). It
validates build completion and returned-ID integrity, not Recall or official
SQL performance. The newly provisioned instance had no seekdb build or official
SIFT250K/Hidden250K dataset, so these timings are diagnostic only.

After adding the committed harness, one additional confirmation pair measured
42,453.733 ms baseline and 41,153.405 ms registered input; both returned the
same top-1 ID and passed ID-integrity checks. That single pair favors
registration by 3.1%, but is within the multi-run spread above and does not
change the decision not to promote it.

## Next profiling target

Input pinning did not shorten the build. The remaining large cost is inside
`cuvsCagraBuild`/graph construction rather than host-buffer setup. Further work
should instrument the fixed NN-Descent build phases and validate against both
official datasets before promoting any algorithm-path change.
