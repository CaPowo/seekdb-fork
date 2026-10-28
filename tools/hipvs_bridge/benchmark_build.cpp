#include "seekdb_cuvs_bridge.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {
constexpr long kRows = 250000;
constexpr long kDimensions = 256;

float value(long row, long column, long seed)
{
  return std::sin(float(row * 31 + column * 17 + seed * 13) * 0.001f);
}
} // namespace

int main()
{
  std::vector<float> base(static_cast<size_t>(kRows * kDimensions));
  for (long row = 0; row < kRows; ++row) {
    for (long column = 0; column < kDimensions; ++column) {
      base[static_cast<size_t>(row * kDimensions + column)] = value(row, column, 0);
    }
  }

  const auto build_start = std::chrono::steady_clock::now();
  void *index = seekdb_cuvs_build(base.data(), kRows, kDimensions);
  const auto build_stop = std::chrono::steady_clock::now();
  if (index == nullptr) { return 2; }

  std::vector<float> query(static_cast<size_t>(kDimensions));
  for (long column = 0; column < kDimensions; ++column) {
    query[static_cast<size_t>(column)] = value(0, column, 1);
  }
  unsigned int ids[10] = {};
  float distances[10] = {};
  const int search_rc = seekdb_cuvs_search(index, query.data(), 1, 10, ids, distances);
  seekdb_cuvs_free(index);
  if (search_rc != 0) { return 3; }

  bool seen[kRows] = {};
  for (unsigned int id : ids) {
    if (id >= static_cast<unsigned int>(kRows) || seen[id]) { return 4; }
    seen[id] = true;
  }
  const double build_ms =
      std::chrono::duration<double, std::milli>(build_stop - build_start).count();
  std::printf("build_ms=%.3f first_id=%u first_distance=%.8f id_integrity=PASS\n",
              build_ms, ids[0], distances[0]);
  return 0;
}
