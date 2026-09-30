// CPU-path tests for the indexed operations and reductions on view_t<T>
// (index sets addressing the slice-space of a view), and for empty index
// sets. The GPU path is covered from Python (test/test_ops.py).
// See test/cpp/CMakeLists.txt for why this target does not define HAS_CUDA.

#include <doctest/doctest.h>

#include <silt/core/tensor.hpp>
#include <silt/core/view.hpp>
#include <silt/op/common.hpp>
#include <silt/op/indexed.hpp>

#include <cmath>
#include <initializer_list>

using silt::index_t;
using silt::tensor_t;
using silt::view_t;

namespace {

//! CPU index set from a list of flat indices.
index_t make_indices(const std::initializer_list<int64_t> values) {
  index_t ind(silt::shape((int)values.size()), silt::CPU);
  int64_t n = 0;
  for (const int64_t v : values)
    ind[n++] = v;
  return ind;
}

//! A [4, 4, 3] tensor holding 10 * cell + channel.
tensor_t<float> make_image() {
  tensor_t<float> t(silt::shape(4, 4, 3), silt::CPU);
  for (int cell = 0; cell < 16; ++cell)
    for (int c = 0; c < 3; ++c)
      t[cell * 3 + c] = 10.0f * cell + c;
  return t;
}

//! One channel of a [4, 4, 3] tensor, as a [4, 4] view.
view_t<float> channel(tensor_t<float>& t, const int c) {
  auto v = t.view<float>();
  v.reshape(4, 4, 3, 1);
  v.index(2, c, 1, 1);
  return v;
}

} // namespace

TEST_SUITE("indexed operations on views") {

  TEST_CASE("indexed_set through a channel view touches only that channel") {
    auto t = make_image();
    silt::indexed_set(channel(t, 1), -1.0f, make_indices({0, 5}));
    for (int cell = 0; cell < 16; ++cell) {
      const bool selected = cell == 0 || cell == 5;
      CHECK(t[cell * 3 + 0] == 10.0f * cell);
      CHECK(t[cell * 3 + 1] == (selected ? -1.0f : 10.0f * cell + 1));
      CHECK(t[cell * 3 + 2] == 10.0f * cell + 2);
    }
  }

  TEST_CASE("indexed scalar arithmetic on a view") {
    auto t = make_image();
    const auto ind = make_indices({2});
    silt::indexed_add(channel(t, 0), 1.0f, ind);
    CHECK(t[2 * 3] == 21.0f);
    silt::indexed_multiply(channel(t, 0), 2.0f, ind);
    CHECK(t[2 * 3] == 42.0f);
    silt::indexed_divide(channel(t, 0), 6.0f, ind);
    CHECK(t[2 * 3] == 7.0f);
  }

  TEST_CASE("indexed binary arithmetic reads rhs at the same logical index") {
    auto t = make_image();
    const auto ind = make_indices({3, 7});
    // channel 0 += channel 2, in place within one tensor
    silt::indexed_add(channel(t, 0), channel(t, 2), ind);
    CHECK(t[3 * 3] == 30.0f + 32.0f);
    CHECK(t[7 * 3] == 70.0f + 72.0f);
    CHECK(t[5 * 3] == 50.0f);
  }

  TEST_CASE("indexed_mix on views interpolates only the selected cells") {
    auto t = make_image();
    silt::indexed_mix(channel(t, 0), channel(t, 2), make_indices({1}), 0.5f);
    CHECK(t[1 * 3] == doctest::Approx(0.5f * 10.0f + 0.5f * 12.0f));
    CHECK(t[2 * 3] == 20.0f);
  }

  TEST_CASE("indexed operations skip out-of-range logical indices") {
    auto t = make_image();
    silt::indexed_set(channel(t, 0), 0.0f, make_indices({-1, 16, 99}));
    for (int cell = 0; cell < 16; ++cell)
      CHECK(t[cell * 3] == 10.0f * cell);
  }

  TEST_CASE("indexed binary operations reject mismatched hosts and sizes") {
    auto t = make_image();
    tensor_t<float> other(silt::shape(4, 4, 1), silt::CPU);
    silt::set(other, 0.0f);
    auto v = other.view<float>();
    v.reshape(4, 4, 1, 1);
    CHECK_NOTHROW(silt::indexed_add(channel(t, 0), v, make_indices({0})));

    auto small = other.view<float>();
    small.reshape(4, 4, 1, 1);
    small.index(0, 0, 1, 2);
    CHECK_THROWS_AS(silt::indexed_add(channel(t, 0), small, make_indices({0})), silt::error::mismatch_size);

    index_t gpu_ind(silt::shape(0), silt::GPU);
    CHECK_THROWS_AS(silt::indexed_add(channel(t, 0), v, gpu_ind), silt::error::mismatch_host);
  }
}

TEST_SUITE("indexed reductions on views") {

  TEST_CASE("reductions through a channel view follow the slice-space") {
    auto t = make_image();
    const auto ind = make_indices({0, 5, 15});
    CHECK(silt::indexed_sum(channel(t, 1), ind) == 1.0f + 51.0f + 151.0f);
    CHECK(silt::indexed_mean(channel(t, 1), ind) == doctest::Approx((1.0f + 51.0f + 151.0f) / 3.0f));
    CHECK(silt::indexed_min(channel(t, 1), ind) == 1.0f);
    CHECK(silt::indexed_max(channel(t, 1), ind) == 151.0f);
  }

  TEST_CASE("argmin / argmax return the logical index into the view") {
    auto t = make_image();
    const auto ind = make_indices({4, 9, 2});
    CHECK(silt::indexed_argmin(channel(t, 2), ind) == 2);
    CHECK(silt::indexed_argmax(channel(t, 2), ind) == 9);
  }
}

TEST_SUITE("empty index sets") {

  TEST_CASE("mutations are no-ops") {
    auto t = make_image();
    const auto empty = make_indices({});
    silt::indexed_set(t, 0.0f, empty);
    silt::indexed_add(t, 1.0f, empty);
    silt::indexed_multiply(t, 2.0f, empty);
    silt::indexed_divide(t, 2.0f, empty);
    silt::indexed_set(channel(t, 0), 0.0f, empty);
    silt::indexed_add(channel(t, 0), channel(t, 1), empty);
    for (int i = 0; i < 48; ++i)
      CHECK(t[i] == 10.0f * (i / 3) + (i % 3));
  }

  TEST_CASE("sum is zero and a floating-point mean is NaN") {
    auto t = make_image();
    const auto empty = make_indices({});
    CHECK(silt::indexed_sum(t, empty) == 0.0f);
    CHECK(silt::indexed_sum(channel(t, 0), empty) == 0.0f);
    CHECK(std::isnan(silt::indexed_mean(t, empty)));
    CHECK(std::isnan(silt::indexed_mean(channel(t, 0), empty)));
  }

  TEST_CASE("reductions without an identity throw") {
    auto t = make_image();
    const auto empty = make_indices({});
    CHECK_THROWS_AS(silt::indexed_min(t, empty), std::invalid_argument);
    CHECK_THROWS_AS(silt::indexed_max(t, empty), std::invalid_argument);
    CHECK_THROWS_AS(silt::indexed_argmin(t, empty), std::invalid_argument);
    CHECK_THROWS_AS(silt::indexed_argmax(channel(t, 0), empty), std::invalid_argument);
  }

  TEST_CASE("an integer mean over an empty set throws") {
    tensor_t<int> t(silt::shape(4), silt::CPU);
    silt::set(t, 1);
    CHECK_THROWS_AS(silt::indexed_mean(t, make_indices({})), std::invalid_argument);
    CHECK(silt::indexed_sum(t, make_indices({})) == 0);
  }
}
