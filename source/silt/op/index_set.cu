#include <silt/core/error.hpp>
#include <silt/core/memory.hpp>
#include <silt/op/indexed.hpp>

#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <algorithm>

namespace silt {

namespace detail {

inline void require_same_host(const index_t a, const index_t b) {
  if (a.host() != b.host())
    throw silt::error::mismatch_host(a.host(), b.host());
}

//! Independent copy on the same host.
inline index_t copy_of(const index_t ind) {
  return ind.copy_to(ind.host());
}

//! Exact-sized copy of the first `count` elements of `scratch`.
inline index_t trim(const index_t scratch, const int64_t count) {
  index_t result(silt::shape((int)count), scratch.host());
  const copy_t kind = scratch.host() == silt::host_t::CPU ? copy_t::HOST_TO_HOST : copy_t::DEVICE_TO_DEVICE;
  silt::device_copy(result.data(), scratch.data(), count * sizeof(int64_t), kind);
  return result;
}

//! Set operation over two sorted ranges into worst-case scratch, trimmed to
//! the exact size. `cpu` / `gpu` take (a0, a1, b0, b1, out) and return the
//! end of the output.
template<typename CpuF, typename GpuF>
index_t combine(const index_t a, const index_t b, const int64_t capacity, CpuF cpu, GpuF gpu) {
  index_t scratch(silt::shape((int)capacity), a.host());
  const int64_t* a0 = a.data();
  const int64_t* b0 = b.data();
  int64_t* out = scratch.data();
  const int64_t* end = nullptr;
  if (a.host() == silt::host_t::CPU)
    end = cpu(a0, a0 + a.elem(), b0, b0 + b.elem(), out);
  else
    end = gpu(a0, a0 + a.elem(), b0, b0 + b.elem(), out);
  return trim(scratch, end - out);
}

} // namespace detail

index_t index_sort_unique(const index_t ind) {

  index_t out = detail::copy_of(ind);
  const int64_t n = (int64_t)out.elem();
  if (n < 2)
    return out;

  int64_t* first = out.data();
  int64_t* last = first + n;
  int64_t* end = last;
  if (out.host() == silt::host_t::CPU) {
    std::sort(first, last);
    end = std::unique(first, last);
  } else {
    thrust::sort(thrust::device, first, last);
    end = thrust::unique(thrust::device, first, last);
  }

  const int64_t count = end - first;
  return count == n ? out : detail::trim(out, count);
}

index_t index_union(const index_t a, const index_t b) {
  detail::require_same_host(a, b);
  if (a.elem() == 0) return detail::copy_of(b);
  if (b.elem() == 0) return detail::copy_of(a);
  return detail::combine(
      a, b, a.elem() + b.elem(),
      [](auto a0, auto a1, auto b0, auto b1, auto out) { return std::set_union(a0, a1, b0, b1, out); },
      [](auto a0, auto a1, auto b0, auto b1, auto out) { return thrust::set_union(thrust::device, a0, a1, b0, b1, out); }
  );
}

index_t index_intersection(const index_t a, const index_t b) {
  detail::require_same_host(a, b);
  if (a.elem() == 0 || b.elem() == 0)
    return index_t(silt::shape(0), a.host());
  return detail::combine(
      a, b, std::min(a.elem(), b.elem()),
      [](auto a0, auto a1, auto b0, auto b1, auto out) { return std::set_intersection(a0, a1, b0, b1, out); },
      [](auto a0, auto a1, auto b0, auto b1, auto out) { return thrust::set_intersection(thrust::device, a0, a1, b0, b1, out); }
  );
}

index_t index_difference(const index_t a, const index_t b) {
  detail::require_same_host(a, b);
  if (a.elem() == 0)
    return index_t(silt::shape(0), a.host());
  if (b.elem() == 0)
    return detail::copy_of(a);
  return detail::combine(
      a, b, a.elem(),
      [](auto a0, auto a1, auto b0, auto b1, auto out) { return std::set_difference(a0, a1, b0, b1, out); },
      [](auto a0, auto a1, auto b0, auto b1, auto out) { return thrust::set_difference(thrust::device, a0, a1, b0, b1, out); }
  );
}

index_t index_symmetric_difference(const index_t a, const index_t b) {
  detail::require_same_host(a, b);
  if (a.elem() == 0) return detail::copy_of(b);
  if (b.elem() == 0) return detail::copy_of(a);
  return detail::combine(
      a, b, a.elem() + b.elem(),
      [](auto a0, auto a1, auto b0, auto b1, auto out) { return std::set_symmetric_difference(a0, a1, b0, b1, out); },
      [](auto a0, auto a1, auto b0, auto b1, auto out) { return thrust::set_symmetric_difference(thrust::device, a0, a1, b0, b1, out); }
  );
}

index_t index_complement(const index_t ind, const silt::shape shape) {

  const int64_t n = shape.elem();
  const int64_t m = (int64_t)ind.elem();
  index_t scratch(silt::shape((int)n), ind.host());
  if (n == 0)
    return scratch;

  int64_t* out = scratch.data();
  int64_t count = 0;

  if (ind.host() == silt::host_t::CPU) {
    // Merge walk: `j` trails the first element of `ind` not below `i`.
    const int64_t* in = ind.data();
    int64_t j = 0;
    for (int64_t i = 0; i < n; ++i) {
      while (j < m && in[j] < i)
        ++j;
      if (j < m && in[j] == i)
        continue;
      out[count++] = i;
    }
  }

  else if (m == 0) {
    thrust::sequence(thrust::device, out, out + n);
    count = n;
  }

  else {
    const thrust::counting_iterator<int64_t> first(0);
    const int64_t* in = ind.data();
    count = thrust::set_difference(thrust::device, first, first + n, in, in + m, out) - out;
  }

  return count == n ? scratch : detail::trim(scratch, count);
}

} // end of namespace silt
