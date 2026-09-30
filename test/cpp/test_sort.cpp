// CPU-path tests for the dense sort. The GPU path is covered from Python
// (test/test_ops.py). See test/cpp/CMakeLists.txt for why this target does
// not define HAS_CUDA.

#include <doctest/doctest.h>

#include <silt/core/tensor.hpp>
#include <silt/op/sort.hpp>

#include <cmath>
#include <initializer_list>
#include <limits>

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

} // namespace

TEST_SUITE("dense sort") {

  TEST_CASE("sorts ascending in place") {
    auto t = make_tensor<float>({3.0f, -1.0f, 2.0f, 2.0f, 0.5f});
    silt::sort(t);
    const float expected[] = {-1.0f, 0.5f, 2.0f, 2.0f, 3.0f};
    for (int i = 0; i < 5; ++i)
      CHECK(t[i] == expected[i]);
  }

  TEST_CASE("sorts int and double") {
    auto a = make_tensor<int>({5, -2, 9, 0});
    silt::sort(a);
    CHECK(a[0] == -2);
    CHECK(a[1] == 0);
    CHECK(a[2] == 5);
    CHECK(a[3] == 9);

    auto b = make_tensor<double>({0.25, -0.5, 0.125});
    silt::sort(b);
    CHECK(b[0] == -0.5);
    CHECK(b[2] == 0.25);
  }

  TEST_CASE("sorts a multi-dimensional tensor in flat order") {
    tensor_t<float> t(silt::shape(2, 3), silt::CPU);
    for (int i = 0; i < 6; ++i)
      t[i] = (float)(5 - i);
    silt::sort(t);
    for (int i = 0; i < 6; ++i)
      CHECK(t[i] == (float)i);
    CHECK(t.shape().elem() == 6);
  }

  TEST_CASE("shares storage with the tensor it was given") {
    auto t = make_tensor<float>({2.0f, 1.0f});
    auto alias = t;
    silt::sort(t);
    CHECK(alias[0] == 1.0f);
  }

  TEST_CASE("empty and single-element tensors are left alone") {
    tensor_t<float> empty(silt::shape(0), silt::CPU);
    CHECK_NOTHROW(silt::sort(empty));
    auto one = make_tensor<float>({4.0f});
    silt::sort(one);
    CHECK(one[0] == 4.0f);
  }

  TEST_CASE("NaNs sort last on the CPU") {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    auto t = make_tensor<float>({2.0f, nan, -1.0f, nan, 0.0f});
    silt::sort(t);
    CHECK(t[0] == -1.0f);
    CHECK(t[1] == 0.0f);
    CHECK(t[2] == 2.0f);
    CHECK(std::isnan(t[3]));
    CHECK(std::isnan(t[4]));
  }

  TEST_CASE("infinities order around the finite values") {
    const float inf = std::numeric_limits<float>::infinity();
    auto t = make_tensor<float>({inf, 1.0f, -inf});
    silt::sort(t);
    CHECK(t[0] == -inf);
    CHECK(t[1] == 1.0f);
    CHECK(t[2] == inf);
  }
}

TEST_SUITE("argsort") {

  TEST_CASE("returns the permutation that sorts the tensor") {
    const auto t = make_tensor<float>({3.0f, -1.0f, 2.0f, 0.5f});
    const auto perm = silt::argsort(t);
    REQUIRE(perm.elem() == 4);
    CHECK(perm[0] == 1);
    CHECK(perm[1] == 3);
    CHECK(perm[2] == 2);
    CHECK(perm[3] == 0);
  }

  TEST_CASE("is stable: equal elements keep their flat order") {
    const auto t = make_tensor<int>({2, 1, 2, 1, 2});
    const auto perm = silt::argsort(t);
    const int64_t expected[] = {1, 3, 0, 2, 4};
    for (int i = 0; i < 5; ++i)
      CHECK(perm[i] == expected[i]);
  }

  TEST_CASE("gathering through the permutation reproduces sort") {
    tensor_t<double> t(silt::shape(50), silt::CPU);
    for (int i = 0; i < 50; ++i)
      t[i] = (double)((i * 17) % 23);
    const auto perm = silt::argsort(t);
    auto sorted = t.copy_to(silt::CPU); // independent copy: sort() is in place
    silt::sort(sorted);
    for (int i = 0; i < 50; ++i)
      CHECK(t[perm[i]] == sorted[i]);
  }

  TEST_CASE("NaNs are ordered last on the CPU") {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const auto t = make_tensor<float>({nan, 1.0f, nan, 0.0f});
    const auto perm = silt::argsort(t);
    CHECK(perm[0] == 3);
    CHECK(perm[1] == 1);
    CHECK(perm[2] == 0);
    CHECK(perm[3] == 2);
  }

  TEST_CASE("the result is an index tensor on the source host") {
    const auto perm = silt::argsort(make_tensor<float>({2.0f, 1.0f}));
    CHECK(perm.host() == silt::CPU);
    CHECK(silt::argsort(tensor_t<float>(silt::shape(0), silt::CPU)).elem() == 0);
  }

  TEST_CASE("the source tensor is left untouched") {
    const auto t = make_tensor<float>({2.0f, 1.0f});
    (void)silt::argsort(t);
    CHECK(t[0] == 2.0f);
    CHECK(t[1] == 1.0f);
  }
}
