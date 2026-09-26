// Ownership and lifetime tests for tensor_t<T> and the polymorphic
// `tensor` wrapper -- the properties that are not reachable from
// Python, because only `tensor`, `view`, `shape` and `slice` are bound
// (see doc/extending.rst). See test/cpp/CMakeLists.txt for why this
// deliberately does not define HAS_CUDA.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <silt/core/tensor.hpp>
#include <silt/op/common.hpp>

using silt::tensor;
using silt::tensor_t;

TEST_SUITE("tensor_t<T> ownership") {

  TEST_CASE("default construction leaves data null but refs non-null") {
    // A default-constructed tensor_t<T> sets _refs = new size_t(0), but
    // deallocate() returns early whenever *_refs == 0 without freeing
    // it, leaking 8 bytes. There's no portable way to assert "no leak"
    // here, so this pins down the state that early return depends on.
    tensor_t<float> t;
    CHECK(t.data() == nullptr);
    CHECK(t._refs != nullptr);
    CHECK(*t._refs == 0);
  }

  TEST_CASE("copy assignment increments the shared refcount") {
    tensor_t<float> a(silt::shape(4), silt::CPU);
    tensor_t<float> b = a;
    CHECK(a.refs() == 2);
    CHECK(b.refs() == 2);
    CHECK(a.data() == b.data());
  }

  TEST_CASE("self copy-assignment does not corrupt the tensor") {
    // operator= calls deallocate() before reading `other`'s members.
    // When `other` *is* `*this` and the refcount is 1, deallocate()
    // frees the data and nulls it out before the (now-dangling)
    // members are copied back onto themselves.
    tensor_t<float> a(silt::shape(4), silt::CPU);
    silt::set(a, 3.0f);

    a = a;  // self copy-assignment

    REQUIRE(a.data() != nullptr);
    CHECK(a.refs() == 1);
    for (size_t i = 0; i < a.elem(); ++i)
      CHECK(a[i] == doctest::Approx(3.0f));
  }

  TEST_CASE("self move-assignment does not corrupt the tensor") {
    // Same root cause as above, via operator=(tensor_t&&).
    tensor_t<float> a(silt::shape(4), silt::CPU);
    silt::set(a, 5.0f);

    a = std::move(a);  // self move-assignment

    REQUIRE(a.data() != nullptr);
    for (size_t i = 0; i < a.elem(); ++i)
      CHECK(a[i] == doctest::Approx(5.0f));
  }

  TEST_CASE("move construction steals the buffer rather than sharing it") {
    // Contrast with the polymorphic `tensor` wrapper below: this
    // constructor already does a true steal, with no refcount change.
    tensor_t<float> a(silt::shape(4), silt::CPU);
    float* raw = a.data();

    tensor_t<float> b(std::move(a));
    CHECK(b.data() == raw);
    CHECK(b.refs() == 1);
    CHECK(a.data() == nullptr);  // moved-from: pointer was stolen, not shared
  }

  TEST_CASE("a non-owning tensor_t<T> has a null refcount") {
    // The non-allocating constructor tensor_t(T*, shape, host) sets
    // _refs = NULL: it does not own the buffer it wraps. Calling refs()
    // on it would dereference that null pointer, so that call is not
    // made here -- this pins down the precondition instead.
    float storage[4] = {0, 0, 0, 0};
    tensor_t<float> t(storage, silt::shape(4), silt::CPU);
    CHECK(t.data() == storage);
    CHECK(t._refs == nullptr);
  }

}

TEST_SUITE("tensor (polymorphic wrapper)") {

  TEST_CASE("copy assignment shares the same tensor_t via refcount") {
    tensor a(silt::FLOAT32, silt::shape(4));
    // Not silt::select(): it instantiates the lambda for every
    // silt::dtype, including RNG, whose set<rng> is never linked in.
    // a's type is already known here.
    silt::set(a.as<float>(), 1.0f);

    tensor b = a;
    CHECK(a.data() == b.data());
    CHECK(a.as<float>().refs() == 2);
  }

  TEST_CASE("move construction clones today rather than stealing") {
    // tensor's move constructor calls rhs.clone(), which allocates a
    // *new* tensor_t<T> (refcount-incrementing the impl, not deep-
    // copying) rather than stealing rhs's impl pointer outright. So a
    // "moved-from" tensor here is left holding a live, still-valid
    // impl, unlike a real move. This pins down today's behaviour, not
    // the desired one: once this steals rhs's impl instead, the second
    // CHECK below should change to expect `a` to be empty.
    tensor a(silt::FLOAT32, silt::shape(4));
    void* original_data = a.data();

    tensor b(std::move(a));
    CHECK(b.data() == original_data);
    CHECK(a.data() == original_data);  // documents the clone, not a requirement
  }

}
