// CPU-path tests for silt::gather / silt::scatter on tensor_t<T> and view_t<T>,
// including index sets addressing the slice-space of a view. The GPU path
// is covered from Python (test/test_ops.py), where index sets can be built.
// See test/cpp/CMakeLists.txt for why this target does not define HAS_CUDA.

#include <doctest/doctest.h>

#include <silt/core/tensor.hpp>
#include <silt/core/view.hpp>
#include <silt/op/common.hpp>
#include <silt/op/indexed.hpp>

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

//! CPU float tensor of `count` consecutive values starting at 0.
tensor_t<float> make_ramp(const int count) {
  tensor_t<float> t(silt::shape(count), silt::CPU);
  for (int i = 0; i < count; ++i)
    t[i] = (float)i;
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

TEST_SUITE("gather") {

  TEST_CASE("gather from a tensor follows the index set order") {
    const auto src = make_ramp(8);
    const auto out = silt::gather(src, make_indices({5, 1, 6}));
    REQUIRE(out.elem() == 3);
    CHECK(out[0] == 5.0f);
    CHECK(out[1] == 1.0f);
    CHECK(out[2] == 6.0f);
  }

  TEST_CASE("gather returns a compact tensor on the source host") {
    const auto out = silt::gather(make_ramp(8), make_indices({0, 7}));
    CHECK(out.host() == silt::CPU);
    CHECK(out.shape().elem() == 2);
  }

  TEST_CASE("gather with out-of-range indices yields zero") {
    const auto src = make_ramp(4);
    const auto out = silt::gather(src, make_indices({-1, 2, 4, 100}));
    REQUIRE(out.elem() == 4);
    CHECK(out[0] == 0.0f);
    CHECK(out[1] == 2.0f);
    CHECK(out[2] == 0.0f);
    CHECK(out[3] == 0.0f);
  }

  TEST_CASE("gather with an empty index set yields an empty tensor") {
    const auto out = silt::gather(make_ramp(4), make_indices({}));
    CHECK(out.elem() == 0);
  }

  TEST_CASE("gather through a channel view addresses the slice-space") {
    // [4, 4, 3] with value 10 * cell + channel; an index set over the
    // [4, 4] cells picks the same cells from every channel.
    tensor_t<float> t(silt::shape(4, 4, 3), silt::CPU);
    for (int cell = 0; cell < 16; ++cell)
      for (int c = 0; c < 3; ++c)
        t[cell * 3 + c] = 10.0f * cell + c;

    const auto ind = make_indices({0, 5, 15});
    for (int c = 0; c < 3; ++c) {
      const auto out = silt::gather(channel(t, c), ind);
      REQUIRE(out.elem() == 3);
      CHECK(out[0] == 0.0f + c);
      CHECK(out[1] == 50.0f + c);
      CHECK(out[2] == 150.0f + c);
    }
  }

  TEST_CASE("gather from a strided view bounds-checks against the view") {
    auto t = make_ramp(8);
    auto v = t.view<float>();
    v.reshape(8, 1, 1, 1);
    v.index(0, 0, 2, 8); // 0, 2, 4, 6 -- 4 elements
    const auto out = silt::gather(v, make_indices({0, 3, 4}));
    REQUIRE(out.elem() == 3);
    CHECK(out[0] == 0.0f);
    CHECK(out[1] == 6.0f);
    CHECK(out[2] == 0.0f); // 4 is outside the 4-element view
  }

  TEST_CASE("gather on mismatched hosts throws") {
    // Zero-element, so no device allocation is needed to build a GPU-hosted index set.
    index_t gpu_ind(silt::shape(0), silt::GPU);
    CHECK_THROWS_AS(silt::gather(make_ramp(4), gpu_ind), silt::error::mismatch_host);
  }

  TEST_CASE("gather works for int and double") {
    tensor_t<int> a(silt::shape(3), silt::CPU);
    tensor_t<double> b(silt::shape(3), silt::CPU);
    for (int i = 0; i < 3; ++i) {
      a[i] = 2 * i;
      b[i] = 0.5 * i;
    }
    const auto ind = make_indices({2, 0});
    CHECK(silt::gather(a, ind)[0] == 4);
    CHECK(silt::gather(b, ind)[0] == doctest::Approx(1.0));
  }
}

TEST_SUITE("scatter") {

  TEST_CASE("scatter into a tensor writes only the indexed cells") {
    auto dst = make_ramp(6);
    auto src = tensor_t<float>(silt::shape(2), silt::CPU);
    src[0] = -1.0f;
    src[1] = -2.0f;
    silt::scatter(dst, src, make_indices({4, 1}));
    CHECK(dst[0] == 0.0f);
    CHECK(dst[1] == -2.0f);
    CHECK(dst[2] == 2.0f);
    CHECK(dst[3] == 3.0f);
    CHECK(dst[4] == -1.0f);
    CHECK(dst[5] == 5.0f);
  }

  TEST_CASE("gather, dense manipulation and scatter round-trip") {
    auto t = make_ramp(8);
    const auto ind = make_indices({1, 3, 5});
    auto dense = silt::gather(t, ind);
    silt::multiply(dense, 10.0f);
    silt::scatter(t, dense, ind);
    CHECK(t[0] == 0.0f);
    CHECK(t[1] == 10.0f);
    CHECK(t[2] == 2.0f);
    CHECK(t[3] == 30.0f);
    CHECK(t[4] == 4.0f);
    CHECK(t[5] == 50.0f);
  }

  TEST_CASE("scatter skips out-of-range indices") {
    auto dst = make_ramp(4);
    auto src = tensor_t<float>(silt::shape(3), silt::CPU);
    silt::set(src, 9.0f);
    silt::scatter(dst, src, make_indices({-1, 2, 4}));
    CHECK(dst[0] == 0.0f);
    CHECK(dst[1] == 1.0f);
    CHECK(dst[2] == 9.0f);
    CHECK(dst[3] == 3.0f);
  }

  TEST_CASE("scatter with an empty index set is a no-op") {
    auto dst = make_ramp(4);
    tensor_t<float> src(silt::shape(0), silt::CPU);
    silt::scatter(dst, src, make_indices({}));
    CHECK(dst[3] == 3.0f);
  }

  TEST_CASE("scatter through a channel view paints one channel") {
    tensor_t<float> t(silt::shape(4, 4, 3), silt::CPU);
    silt::set(t, 0.0f);

    const auto ind = make_indices({0, 5, 15});
    tensor_t<float> color(silt::shape(3), silt::CPU);
    const float rgb[3] = {0.25f, 0.5f, 0.75f};
    for (int c = 0; c < 3; ++c) {
      silt::set(color, rgb[c]);
      silt::scatter(channel(t, c), color, ind);
    }

    for (int cell = 0; cell < 16; ++cell) {
      const bool selected = cell == 0 || cell == 5 || cell == 15;
      for (int c = 0; c < 3; ++c)
        CHECK(t[cell * 3 + c] == (selected ? rgb[c] : 0.0f));
    }
  }

  TEST_CASE("scatter with a mismatched source size throws") {
    auto dst = make_ramp(4);
    auto src = make_ramp(3);
    CHECK_THROWS_AS(silt::scatter(dst, src, make_indices({0, 1})), silt::error::mismatch_size);
  }

  TEST_CASE("scatter on mismatched hosts throws") {
    auto dst = make_ramp(4);
    auto src = make_ramp(2);
    // Zero-element, so no device allocation is needed to build a GPU-hosted index set.
    index_t gpu_ind(silt::shape(0), silt::GPU);
    CHECK_THROWS_AS(silt::scatter(dst, src, gpu_ind), silt::error::mismatch_host);
  }
}
