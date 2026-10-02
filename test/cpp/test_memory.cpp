// CPU-path tests for the allocation ledger. The GPU path is covered from
// Python (test/test_memory.py).
// See test/cpp/CMakeLists.txt for why this target does not define HAS_CUDA.

#include <doctest/doctest.h>

#include <silt/core/memory.hpp>
#include <silt/core/tensor.hpp>

using silt::tensor_t;

TEST_SUITE("memory") {

  TEST_CASE("a cpu tensor is counted while it lives") {
    const auto before = silt::memory_usage();
    {
      tensor_t<float> t(silt::shape(16, 16), silt::CPU);
      const auto during = silt::memory_usage();
      CHECK(during.cpu_bytes == before.cpu_bytes + 16 * 16 * sizeof(float));
      CHECK(during.cpu_allocations == before.cpu_allocations + 1);
      CHECK(during.gpu_bytes == before.gpu_bytes);
    }
    const auto after = silt::memory_usage();
    CHECK(after.cpu_bytes == before.cpu_bytes);
    CHECK(after.cpu_allocations == before.cpu_allocations);
  }

  TEST_CASE("shared copies do not double count") {
    const auto before = silt::memory_usage();
    tensor_t<double> a(silt::shape(32), silt::CPU);
    {
      tensor_t<double> b = a;
      CHECK(silt::memory_usage().cpu_bytes == before.cpu_bytes + 32 * sizeof(double));
    }
    CHECK(silt::memory_usage().cpu_bytes == before.cpu_bytes + 32 * sizeof(double));
  }

  TEST_CASE("copy_to allocates a second buffer") {
    tensor_t<int> a(silt::shape(8), silt::CPU);
    const auto before = silt::memory_usage();
    {
      const auto b = a.copy_to(silt::CPU);
      CHECK(silt::memory_usage().cpu_bytes == before.cpu_bytes + 8 * sizeof(int));
    }
    CHECK(silt::memory_usage().cpu_bytes == before.cpu_bytes);
  }

  TEST_CASE("zero-element tensors hold no memory") {
    const auto before = silt::memory_usage();
    tensor_t<float> t(silt::shape(0), silt::CPU);
    CHECK(silt::memory_usage().cpu_bytes == before.cpu_bytes);
    CHECK(silt::memory_usage().cpu_allocations == before.cpu_allocations);
  }

  TEST_CASE("the peak keeps the high-water mark until reset") {
    silt::memory_reset_peak();
    const auto base = silt::memory_usage();
    {
      tensor_t<float> t(silt::shape(1024), silt::CPU);
    }
    const auto after = silt::memory_usage();
    CHECK(after.cpu_bytes == base.cpu_bytes);
    CHECK(after.cpu_peak == base.cpu_bytes + 1024 * sizeof(float));
    silt::memory_reset_peak();
    CHECK(silt::memory_usage().cpu_peak == base.cpu_bytes);
  }

  TEST_CASE("the ledger id is stable and nonzero") {
    CHECK(silt::memory_usage().id != 0);
    CHECK(silt::memory_usage().id == silt::memory_usage().id);
  }
}
