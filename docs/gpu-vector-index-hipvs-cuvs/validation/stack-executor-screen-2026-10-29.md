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

## Snapshot transfer optimization

After CAGRA serialization to a temporary file, the adaptor copies the graph
file into the storage stream. The previous loop used 64 KiB read/write chunks;
the retained change uses 4 MiB chunks for both snapshot write and reload. Stage
timings were:

| Chunk | Total build + first query (ms) | CAGRA build + temp serialize (ms) | Snapshot stream write (ms) |
|---:|---:|---:|---:|
| 64 KiB | 60,024 | 43,586 | 9,440 |
| 4 MiB, run 1 | 58,355 | 46,276 | 5,079 |
| 4 MiB, run 2 | 59,783 | 44,286 | 8,640 |

The two 4 MiB runs averaged 6.86 s in snapshot output versus 9.44 s in the
single 64 KiB sample (about 27% less in that stage). Total build+first-query
time averaged 59.07 s versus 60.02 s; because the old-chunk sample count is one
and CAGRA construction varies, this is a modest diagnostic result rather than
a claimed general build-speedup. Graph bytes remained 384,001,713 and result
row counts/duplicate checks passed. A server restart then loaded the saved
4 MiB-chunk snapshot and returned 10 IDs through one traced GPU CAGRA serve.

### Avoid duplicate FP32 vector payload in snapshot V3

The V2 snapshot wrote the raw FP32 rows and then wrote the serialized CAGRA
graph, which already embeds its dataset. V3 retains the ID list and CAGRA graph
but omits the second copy of the vectors. On deserialization it recovers the
FP32 rows from `cuvsCagraIndexGetDataset()` into the adaptor's staging buffer.
The reader still accepts V2 snapshots, so existing persisted indexes remain
loadable. This changes only the private snapshot format; SQL and FP32 behavior
are unchanged.

The two V3 runs used the same 250,000 x 256 synthetic FP32 vectors, fixed
`niter=100`, graph degree 128, intermediate degree 256, and 4,096 single-query
Top-10 plus Top-100 statements as the V2 4 MiB runs:

| Format | Run | Build + first query (ms) | CAGRA build + temp serialize (ms) | Snapshot stream write (ms) | Top-10 / Top-100 QPS | Recall@10 / @100 |
|---|---:|---:|---:|---:|---:|---:|
| V2, 4 MiB | 1 | 58,355 | 46,276 | 5,079 | 197.27 / 189.65 | 0.937012 / 0.865330 |
| V2, 4 MiB | 2 | 59,783 | 44,286 | 8,640 | 196.66 / 185.91 | 0.943018 / 0.865740 |
| V3 | 1 | 53,912 | 45,808 | 298 | 251.35 / 245.02 | 0.948022 / 0.868337 |
| V3 | 2 | 55,462 | 47,785 | 300 | 241.88 / 242.19 | 0.945996 / 0.869231 |

Across these two runs, V3's build+first-query mean was 54.69 s versus 59.07 s
for V2 4 MiB (7.4% lower); snapshot stream output fell from 6.86 s to 0.30 s
(95.6% lower). CAGRA build plus temporary graph serialization varied and did
not improve in these samples, so the end-to-end reduction is attributable to
removing redundant snapshot output, not a faster build stage. The 250,000 x 256 raw-vector
copy omitted from the outer payload is 256,000,000 bytes (about 244 MiB).
V3 returned all 409,600 Top-100 rows with no duplicate-result queries in both
runs.

Persistence checks on the GPU instance:

* V3 was loaded after a full server restart. The table retained 250,000 rows;
  its first 10-result query returned 10 distinct IDs and emitted `cuvs_serve`.
  The server logged `deserialized GPU CAGRA snapshot(... graph_bytes=384001713)`.
* A prior V2 index (`items_snapshotchunk_final4mb100`) was also reopened after
  restart, returned 10 distinct IDs, and emitted `cuvs_serve`.
* The bridge C API persistence test wrote and reloaded a 10,000 x 32 FP32
  graph. Exact vector restoration, top-1 ID 0, and distance 0 passed; wrong
  shape checks (9,999 x 32 and 10,000 x 33) were rejected.

The exact benchmark invocation for each V3 sample was:

```bash
bash /workspace/amd-cagra-v5-validation/new-image/run_variant_256_nq4096.sh \
  snapshot_v3_test100 NN_DESCENT 128 256 100
bash /workspace/amd-cagra-v5-validation/new-image/run_variant_256_nq4096.sh \
  snapshot_v3_test100_r2 NN_DESCENT 128 256 100
```

These numbers are still synthetic GPU-instance diagnostics, not official
SIFT250K/Hidden250K scores. The benchmark samples are small; the storage-format
change and compatibility tests pass, but no official score or quality-gate
claim follows from this screen.

The checked output artifacts were revalidated after both runs (validation
command exit status 0). Each of the four runs had 8,193 `cuvs_serve` traces,
40,960 Top-10 rows, 409,600 Top-100 rows, and zero duplicate-result queries.
The captured recall and timing summaries were:

```text
V2 snapshotchunk_4mb100: build+first=58355 ms; top10=20763 ms/QPS 197.27; top100=21598 ms/QPS 189.65; Recall@10=0.937012; Recall@100=0.865330; exit_status=0
V2 snapshotchunk_final4mb100: build+first=59783 ms; top10=20828 ms/QPS 196.66; top100=22032 ms/QPS 185.91; Recall@10=0.943018; Recall@100=0.865740; exit_status=0
V3 snapshot_v3_test100: build+first=53912 ms; top10=16296 ms/QPS 251.35; top100=16717 ms/QPS 245.02; Recall@10=0.948022; Recall@100=0.868337; exit_status=0
V3 snapshot_v3_test100_r2: build+first=55462 ms; top10=16934 ms/QPS 241.88; top100=16912 ms/QPS 242.19; Recall@10=0.945996; Recall@100=0.869231; exit_status=0
```

## FP16 probe (discarded)

An internal FP16 CAGRA build/query probe reduced the serialized graph from
384,001,713 to 256,001,713 bytes and measured 54.2–54.5 s build+first-query
time on the same synthetic screen. Recall varied between 0.938–0.946 at @10
and 0.864–0.867 at @100, with no duplicate IDs. However, the fixed hipVS
26.03 `cuvsCagraDeserialize` rejected FP16 graph files with `Unsupported dtype
in file`; FP16 snapshot round-trip therefore failed. The FP16 code was removed
and is not part of the candidate. The shipped path remains FP32 and retains
the snapshot format's load behavior.

## Lower-iteration / wider-search probe (not retained)

I also tested a build/query tradeoff using FP32 data: `nn_descent_niter=50`
with the fixed SQL unchanged, varying only the bridge's internal CAGRA
`search_width`. It can recover part of the recall lost by a cheaper graph, but
the repeated @10 result still fluctuated below the official 0.95 SIFT floor on
this synthetic screen, so none of these settings was promoted:

| `search_width` | Build + first query (ms) | Top-10 / Top-100 QPS | Recall@10 / @100 |
|---:|---:|---:|---:|
| 8 | 40,366 | 236.53 / 229.66 | 0.940015 / 0.867737 |
| 16 | 39,270 | 367.26 / 355.34 | 0.949976 / 0.868740 |
| 32, run 1 | 44,303 | 359.24 / 327.58 | 0.950000 / 0.875525 |
| 32, run 2 | 45,584 | 379.40 / 382.98 | 0.943970 / 0.876428 |

All completed runs returned the expected row counts with no duplicate IDs and
8,193 traced GPU serves. `search_width=64` did not complete even its first
query within three minutes and was stopped. These measurements motivate
testing with the official datasets, but do not justify changing the frozen
FP32 build configuration in this branch.

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
