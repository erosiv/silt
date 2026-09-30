// CPU-path tests for silt::histogram on tensor_t<T> and view_t<T>. The GPU
// path is covered from Python (test/test_ops.py).
// See test/cpp/CMakeLists.txt for why this target does not define HAS_CUDA.

#include <doctest/doctest.h>

#include <silt/core/tensor.hpp>
#include <silt/core/view.hpp>
#include <silt/op/histogram.hpp>

#include <cmath>
#include <initializer_list>
#include <limits>
#include <vector>

using silt::tensor_t;

namespace {

template<typename T>
tensor_t<T> make_tensor(const std::initializer_list<T> values) {
  tensor_t<T> t(silt::shape((int)values.size()), silt::CPU);
  size_t n = 0;
  for (const T v : values)
    t[n++] = v;
  return t;
}

std::vector<int> to_vector(const tensor_t<int>& hist) {
  std::vector<int> out;
  for (size_t i = 0; i < hist.elem(); ++i)
    out.push_back(hist[i]);
  return out;
}

using vec = std::vector<int>;

} // namespace

TEST_SUITE("histogram") {

  TEST_CASE("counts elements into equal-width bins") {
    const auto data = make_tensor<float>({0.5f, 1.5f, 1.6f, 2.5f, 3.5f, 3.9f});
    CHECK(to_vector(silt::histogram(data, 4, 0.0f, 4.0f)) == vec{1, 2, 1, 2});
  }

  TEST_CASE("the range is [lo, hi]: lo is in the first bin, hi in the last") {
    const auto data = make_tensor<float>({0.0f, 1.0f, 2.0f, 3.0f, 4.0f});
    CHECK(to_vector(silt::histogram(data, 4, 0.0f, 4.0f)) == vec{1, 1, 1, 2});
  }

  TEST_CASE("bin edges belong to the upper bin") {
    const auto data = make_tensor<float>({1.0f, 2.0f});
    CHECK(to_vector(silt::histogram(data, 4, 0.0f, 4.0f)) == vec{0, 1, 1, 0});
  }

  TEST_CASE("elements outside the range and NaNs are not counted") {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const auto data = make_tensor<float>({-1.0f, 0.5f, nan, 5.0f, 3.5f});
    CHECK(to_vector(silt::histogram(data, 4, 0.0f, 4.0f)) == vec{1, 0, 0, 1});
  }

  TEST_CASE("a single bin counts everything in range") {
    const auto data = make_tensor<float>({0.0f, 2.0f, 4.0f, 9.0f});
    CHECK(to_vector(silt::histogram(data, 1, 0.0f, 4.0f)) == vec{3});
  }

  TEST_CASE("int and double data") {
    CHECK(to_vector(silt::histogram(make_tensor<int>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}), 5, 0, 10)) == vec{2, 2, 2, 2, 3});
    CHECK(to_vector(silt::histogram(make_tensor<double>({0.1, 0.2, 0.9}), 2, 0.0, 1.0)) == vec{2, 1});
  }

  TEST_CASE("an empty tensor gives zero counts") {
    tensor_t<float> empty(silt::shape(0), silt::CPU);
    CHECK(to_vector(silt::histogram(empty, 3, 0.0f, 1.0f)) == vec{0, 0, 0});
  }

  TEST_CASE("the counts sum to the number of in-range elements") {
    tensor_t<float> data(silt::shape(1000), silt::CPU);
    for (int i = 0; i < 1000; ++i)
      data[i] = (float)((i * 37) % 200) * 0.05f; // 0 .. 9.95
    const auto hist = silt::histogram(data, 16, 0.0f, 8.0f);
    int total = 0;
    int expected = 0;
    for (size_t i = 0; i < hist.elem(); ++i)
      total += hist[i];
    for (int i = 0; i < 1000; ++i)
      expected += data[i] <= 8.0f ? 1 : 0;
    CHECK(total == expected);
  }

  TEST_CASE("a channel view is histogrammed in its slice-space") {
    // [2, 2, 2]: channel 0 holds {0, 2, 4, 6}, channel 1 holds {1, 3, 5, 7}.
    tensor_t<float> t(silt::shape(2, 2, 2), silt::CPU);
    for (int i = 0; i < 8; ++i)
      t[i] = (float)i;
    auto v = t.view<float>();
    v.reshape(2, 2, 2, 1);
    v.index(2, 1, 1, 1);
    CHECK(to_vector(silt::histogram(v, 2, 0.0f, 8.0f)) == vec{2, 2});
    CHECK(to_vector(silt::histogram(v, 4, 0.0f, 8.0f)) == vec{1, 1, 1, 1});
  }

  TEST_CASE("an invalid bin count or range throws") {
    const auto data = make_tensor<float>({1.0f});
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    CHECK_THROWS_AS(silt::histogram(data, 0, 0.0f, 1.0f), std::invalid_argument);
    CHECK_THROWS_AS(silt::histogram(data, -3, 0.0f, 1.0f), std::invalid_argument);
    CHECK_THROWS_AS(silt::histogram(data, 4, 1.0f, 1.0f), std::invalid_argument);
    CHECK_THROWS_AS(silt::histogram(data, 4, 2.0f, 1.0f), std::invalid_argument);
    CHECK_THROWS_AS(silt::histogram(data, 4, 0.0f, inf), std::invalid_argument);
    CHECK_THROWS_AS(silt::histogram(data, 4, nan, 1.0f), std::invalid_argument);
  }
}
