// CPU-path tests for index-set algebra (union, intersection, difference,
// symmetric difference, complement, sort_unique). The GPU path shares the
// public API and is covered from Python (test/test_ops.py).
// See test/cpp/CMakeLists.txt for why this target does not define HAS_CUDA.

#include <doctest/doctest.h>

#include <silt/core/tensor.hpp>
#include <silt/op/indexed.hpp>

#include <algorithm>
#include <initializer_list>
#include <iterator>
#include <set>
#include <vector>

using silt::index_t;

namespace {

//! CPU index set from a list of flat indices, in the given order.
index_t make_indices(const std::initializer_list<int64_t> values) {
  index_t ind(silt::shape((int)values.size()), silt::CPU);
  int64_t n = 0;
  for (const int64_t v : values)
    ind[n++] = v;
  return ind;
}

std::vector<int64_t> to_vector(const index_t& ind) {
  std::vector<int64_t> out;
  for (size_t i = 0; i < ind.elem(); ++i)
    out.push_back(ind[i]);
  return out;
}

using vec = std::vector<int64_t>;

} // namespace

TEST_SUITE("index set algebra") {

  TEST_CASE("union merges two overlapping sets") {
    CHECK(to_vector(silt::index_union(make_indices({1, 3, 5, 7}), make_indices({3, 4, 7, 9}))) == vec{1, 3, 4, 5, 7, 9});
  }

  TEST_CASE("intersection keeps only the shared indices") {
    CHECK(to_vector(silt::index_intersection(make_indices({1, 3, 5, 7}), make_indices({3, 4, 7, 9}))) == vec{3, 7});
  }

  TEST_CASE("difference removes the second set from the first") {
    CHECK(to_vector(silt::index_difference(make_indices({1, 3, 5, 7}), make_indices({3, 4, 7, 9}))) == vec{1, 5});
    CHECK(to_vector(silt::index_difference(make_indices({3, 4, 7, 9}), make_indices({1, 3, 5, 7}))) == vec{4, 9});
  }

  TEST_CASE("symmetric difference keeps indices in exactly one set") {
    CHECK(to_vector(silt::index_symmetric_difference(make_indices({1, 3, 5, 7}), make_indices({3, 4, 7, 9}))) == vec{1, 4, 5, 9});
  }

  TEST_CASE("disjoint and identical sets") {
    const auto a = make_indices({0, 2, 4});
    const auto b = make_indices({1, 3, 5});
    CHECK(to_vector(silt::index_union(a, b)) == vec{0, 1, 2, 3, 4, 5});
    CHECK(silt::index_intersection(a, b).elem() == 0);
    CHECK(to_vector(silt::index_difference(a, b)) == vec{0, 2, 4});
    CHECK(to_vector(silt::index_symmetric_difference(a, a)).empty());
    CHECK(silt::index_difference(a, a).elem() == 0);
    CHECK(to_vector(silt::index_intersection(a, a)) == vec{0, 2, 4});
    CHECK(to_vector(silt::index_union(a, a)) == vec{0, 2, 4});
  }

  TEST_CASE("empty operands") {
    const auto a = make_indices({2, 4});
    const auto e = make_indices({});
    CHECK(to_vector(silt::index_union(a, e)) == vec{2, 4});
    CHECK(to_vector(silt::index_union(e, a)) == vec{2, 4});
    CHECK(silt::index_intersection(a, e).elem() == 0);
    CHECK(silt::index_intersection(e, a).elem() == 0);
    CHECK(to_vector(silt::index_difference(a, e)) == vec{2, 4});
    CHECK(silt::index_difference(e, a).elem() == 0);
    CHECK(to_vector(silt::index_symmetric_difference(a, e)) == vec{2, 4});
    CHECK(silt::index_union(e, e).elem() == 0);
  }

  TEST_CASE("results are independent copies of the operands") {
    const auto a = make_indices({2, 4});
    const auto out = silt::index_union(a, make_indices({}));
    CHECK(out.data() != a.data());
  }

  TEST_CASE("operations match std::set on a larger example") {
    std::set<int64_t> sa, sb;
    for (int64_t i = 0; i < 200; ++i) {
      if (i % 3 == 0) sa.insert(i);
      if (i % 5 == 0 || i > 150) sb.insert(i);
    }
    index_t ia(silt::shape((int)sa.size()), silt::CPU);
    index_t ib(silt::shape((int)sb.size()), silt::CPU);
    int64_t n = 0;
    for (const int64_t v : sa) ia[n++] = v;
    n = 0;
    for (const int64_t v : sb) ib[n++] = v;

    vec expect;
    std::set_union(sa.begin(), sa.end(), sb.begin(), sb.end(), std::back_inserter(expect));
    CHECK(to_vector(silt::index_union(ia, ib)) == expect);
    expect.clear();
    std::set_intersection(sa.begin(), sa.end(), sb.begin(), sb.end(), std::back_inserter(expect));
    CHECK(to_vector(silt::index_intersection(ia, ib)) == expect);
    expect.clear();
    std::set_difference(sa.begin(), sa.end(), sb.begin(), sb.end(), std::back_inserter(expect));
    CHECK(to_vector(silt::index_difference(ia, ib)) == expect);
    expect.clear();
    std::set_symmetric_difference(sa.begin(), sa.end(), sb.begin(), sb.end(), std::back_inserter(expect));
    CHECK(to_vector(silt::index_symmetric_difference(ia, ib)) == expect);
  }

  TEST_CASE("operands on different hosts throw") {
    // Zero-element, so no device allocation is needed to build a GPU-hosted index set.
    index_t gpu(silt::shape(0), silt::GPU);
    const auto cpu = make_indices({1});
    CHECK_THROWS_AS(silt::index_union(cpu, gpu), silt::error::mismatch_host);
    CHECK_THROWS_AS(silt::index_intersection(cpu, gpu), silt::error::mismatch_host);
    CHECK_THROWS_AS(silt::index_difference(cpu, gpu), silt::error::mismatch_host);
    CHECK_THROWS_AS(silt::index_symmetric_difference(cpu, gpu), silt::error::mismatch_host);
  }
}

TEST_SUITE("index_complement") {

  TEST_CASE("complement is the difference from the full domain") {
    const silt::shape shape(4, 4);
    CHECK(to_vector(silt::index_complement(make_indices({0, 5, 15}), shape)) == vec{1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14});
  }

  TEST_CASE("complement of an empty set is the whole domain") {
    CHECK(to_vector(silt::index_complement(make_indices({}), silt::shape(5))) == vec{0, 1, 2, 3, 4});
  }

  TEST_CASE("complement of the whole domain is empty") {
    CHECK(silt::index_complement(make_indices({0, 1, 2}), silt::shape(3)).elem() == 0);
  }

  TEST_CASE("indices outside the domain are ignored") {
    CHECK(to_vector(silt::index_complement(make_indices({-3, 1, 9, 12}), silt::shape(4))) == vec{0, 2, 3});
  }

  TEST_CASE("complement is an involution over the domain") {
    const silt::shape shape(4, 4);
    const auto a = make_indices({2, 3, 8, 13});
    CHECK(to_vector(silt::index_complement(silt::index_complement(a, shape), shape)) == to_vector(a));
  }

  TEST_CASE("an empty domain has an empty complement") {
    CHECK(silt::index_complement(make_indices({}), silt::shape(0)).elem() == 0);
  }
}

TEST_SUITE("index_sort_unique") {

  TEST_CASE("sorts and removes duplicates") {
    CHECK(to_vector(silt::index_sort_unique(make_indices({5, 1, 5, 3, 1, 9, 0}))) == vec{0, 1, 3, 5, 9});
  }

  TEST_CASE("an already sorted, unique set is unchanged") {
    CHECK(to_vector(silt::index_sort_unique(make_indices({1, 2, 3}))) == vec{1, 2, 3});
  }

  TEST_CASE("trivially small sets") {
    CHECK(silt::index_sort_unique(make_indices({})).elem() == 0);
    CHECK(to_vector(silt::index_sort_unique(make_indices({7}))) == vec{7});
    CHECK(to_vector(silt::index_sort_unique(make_indices({4, 4, 4}))) == vec{4});
  }

  TEST_CASE("the input is left untouched") {
    const auto a = make_indices({3, 1, 3});
    (void)silt::index_sort_unique(a);
    CHECK(to_vector(a) == vec{3, 1, 3});
  }

  TEST_CASE("its output feeds the set operations") {
    const auto a = silt::index_sort_unique(make_indices({9, 3, 3, 1}));
    const auto b = silt::index_sort_unique(make_indices({3, 9, 4}));
    CHECK(to_vector(silt::index_intersection(a, b)) == vec{3, 9});
  }
}
