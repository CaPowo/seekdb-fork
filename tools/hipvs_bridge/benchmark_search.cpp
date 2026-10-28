#include "seekdb_cuvs_bridge.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <vector>

namespace {
constexpr long kRows = 250000;
constexpr long kDimensions = 256;
constexpr long kQueryCount = 4096;
constexpr long kWarmupCount = 32;

float value(long row, long column, long seed)
{
  return std::sin(float(row * 31 + column * 17 + seed * 13) * 0.001f);
}

void fill_vectors(std::vector<float> &vectors, long rows, long seed)
{
  for (long row = 0; row < rows; ++row) {
    for (long column = 0; column < kDimensions; ++column) {
      vectors[static_cast<size_t>(row * kDimensions + column)] = value(row, column, seed);
    }
  }
}
} // namespace

int main(int argc, char **argv)
{
  if (argc != 3) { return 2; }
  std::vector<float> base(static_cast<size_t>(kRows * kDimensions));
  fill_vectors(base, kRows, 0);

  void *index = nullptr;
  if (std::strcmp(argv[1], "build") == 0) {
    index = seekdb_cuvs_build(base.data(), kRows, kDimensions);
    if (index == nullptr || seekdb_cuvs_serialize(index, argv[2]) != 0) { return 3; }
    seekdb_cuvs_free(index);
    std::puts("build_snapshot=PASS");
    return 0;
  }

  index = seekdb_cuvs_deserialize(argv[2], kRows, kDimensions);
  if (index == nullptr) { return 4; }
  std::vector<float> queries(static_cast<size_t>(kQueryCount * kDimensions));
  fill_vectors(queries, kQueryCount, 1);
  for (long topk : {10, 100}) {
    std::vector<unsigned int> ids(static_cast<size_t>(topk));
    std::vector<float> distances(static_cast<size_t>(topk));
    for (long i = 0; i < kWarmupCount; ++i) {
      if (seekdb_cuvs_search(index, queries.data() + (i % kQueryCount) * kDimensions,
                             1, topk, ids.data(), distances.data()) != 0) { return 5; }
    }
    std::vector<unsigned int> outputs(static_cast<size_t>(kQueryCount * topk));
    const auto start = std::chrono::steady_clock::now();
    for (long i = 0; i < kQueryCount; ++i) {
      if (seekdb_cuvs_search(index, queries.data() + i * kDimensions, 1, topk,
                             ids.data(), distances.data()) != 0) { return 6; }
      std::memcpy(outputs.data() + i * topk, ids.data(),
                  static_cast<size_t>(topk) * sizeof(unsigned int));
    }
    const auto stop = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(stop - start).count();
    uint64_t hash = 1469598103934665603ULL;
    size_t bad_ids = 0;
    size_t duplicate_rows = 0;
    for (long i = 0; i < kQueryCount; ++i) {
      bool duplicate = false;
      for (long a = 0; a < topk; ++a) {
        const unsigned int id = outputs[static_cast<size_t>(i * topk + a)];
        if (id >= static_cast<unsigned int>(kRows)) { ++bad_ids; }
        hash = (hash ^ id) * 1099511628211ULL;
        for (long b = 0; b < a; ++b) {
          if (outputs[static_cast<size_t>(i * topk + b)] == id) {
            duplicate = true;
            break;
          }
        }
      }
      if (duplicate) { ++duplicate_rows; }
    }
    if (bad_ids != 0 || duplicate_rows != 0) { return 7; }
    std::printf("topk=%ld nq=%ld elapsed_ms=%.3f qps=%.3f hash=%016llx duplicate_rows=%zu\n",
                topk, kQueryCount, elapsed_ms, 1000.0 * kQueryCount / elapsed_ms,
                (unsigned long long)hash, duplicate_rows);
  }
  seekdb_cuvs_free(index);
  return 0;
}
