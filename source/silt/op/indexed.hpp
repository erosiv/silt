#pragma once

#include <silt/silt.hpp>
#include <silt/core/shape.hpp>
#include <silt/core/slice.hpp>
#include <silt/core/tensor.hpp>
#include <silt/core/types.hpp>

#include <limits>
#include <type_traits>

namespace silt {

//
// Index Sets
//
//  A compacted list of flat tensor indices.
//

using index_t = tensor_t<int64_t>;

//
// Generic Compaction: Predicate -> Index Set
//
//  F must be a device-callable functor:
//
//    GPU_ENABLE bool operator()(const int64_t n) const;
//
template<typename F>
index_t make_index_set(const silt::shape shape, F predicate);

//
// Selector Predicates
//

//! All cells within `radius` of `center` (Euclidean, in cell units).
struct radius_predicate {
  silt::shape shape;
  silt::vec2 center;
  float radius;
  GPU_ENABLE bool operator()(const int64_t n) const {
    const silt::vec2 pos = shape.unflatten(n);
    return glm::length(pos - center) < radius;
  }
};

//! All cells in the half-open box [lo, hi).
struct box_predicate {
  silt::shape shape;
  silt::vec2 lo;
  silt::vec2 hi;
  GPU_ENABLE bool operator()(const int64_t n) const {
    const silt::vec2 pos = shape.unflatten(n);
    return pos.x >= lo.x && pos.x < hi.x && pos.y >= lo.y && pos.y < hi.y;
  }
};

//! Point-in-polygon (even-odd / ray-casting rule).
struct polygon_predicate {
  silt::shape shape;
  const silt::vec2* vertices;
  int count;
  GPU_ENABLE bool operator()(const int64_t n) const {
    const silt::vec2 pos = shape.unflatten(n);
    bool inside = false;
    for (int i = 0, j = count - 1; i < count; j = i++) {
      const silt::vec2 a = vertices[i];
      const silt::vec2 b = vertices[j];
      if ((a.y > pos.y) != (b.y > pos.y) &&
          (pos.x < (b.x - a.x) * (pos.y - a.y) / (b.y - a.y) + a.x)) {
        inside = !inside;
      }
    }
    return inside;
  }
};

EXPORT_SHARED index_t index_radius(const silt::shape shape, const silt::vec2 center, const float radius);
EXPORT_SHARED index_t index_box(const silt::shape shape, const silt::vec2 lo, const silt::vec2 hi);
EXPORT_SHARED index_t index_polygon(const silt::shape shape, const tensor_t<silt::vec2>& vertices);

//! Flat indices of every cell a slice (offset/stride/extent per dimension)
//! visits, in slice order. Enumeration, not compaction -- every candidate
//! is included, so this needs no predicate.
EXPORT_SHARED index_t index_slice(const silt::slice& slice);

//! +infinity for T (numeric_limits::max() for an integral T, which has no infinity).
template<typename T>
constexpr T positive_infinity() {
  if constexpr (std::is_floating_point_v<T>)
    return std::numeric_limits<T>::infinity();
  else
    return std::numeric_limits<T>::max();
}

//! -infinity for T (numeric_limits::min() for an integral T, which has no infinity).
template<typename T>
constexpr T negative_infinity() {
  if constexpr (std::is_floating_point_v<T>)
    return -std::numeric_limits<T>::infinity();
  else
    return std::numeric_limits<T>::min();
}

//! All cells where lo <= data[n] <= hi (closed interval).
template<typename T>
struct range_predicate {
  const T* data;
  T lo;
  T hi;
  GPU_ENABLE bool operator()(const int64_t n) const {
    const T v = data[n];
    return v >= lo && v <= hi;
  }
};

//! Selects by value rather than position: lo <= data[n] <= hi.
template<typename T>
index_t index_range(const tensor_t<T>& data, const T lo, const T hi);

//! Alias for index_range(data, value, +infinity) -- data[n] >= value.
template<typename T>
index_t index_greater(const tensor_t<T>& data, const T value);

//! Alias for index_range(data, -infinity, value) -- data[n] <= value.
template<typename T>
index_t index_lesser(const tensor_t<T>& data, const T value);

//! Alias for index_range(data, value, value) -- data[n] == value.
template<typename T>
index_t index_match(const tensor_t<T>& data, const T value);

//
// Indexed Operations
//

//! lhs[i] = rhs, for i in ind.
template<typename T>
void indexed_set(tensor_t<T> lhs, const T rhs, const index_t ind);

//! lhs[i] += rhs, for i in ind.
template<typename T>
void indexed_add(tensor_t<T> lhs, const T rhs, const index_t ind);

//! lhs[i] += rhs[i], for i in ind.
template<typename T>
void indexed_add(tensor_t<T> lhs, const tensor_t<T> rhs, const index_t ind);

//! lhs[i] *= rhs, for i in ind.
template<typename T>
void indexed_multiply(tensor_t<T> lhs, const T rhs, const index_t ind);

//! lhs[i] *= rhs[i], for i in ind.
template<typename T>
void indexed_multiply(tensor_t<T> lhs, const tensor_t<T> rhs, const index_t ind);

//! lhs[i] /= rhs, for i in ind.
template<typename T>
void indexed_divide(tensor_t<T> lhs, const T rhs, const index_t ind);

//! lhs[i] /= rhs[i], for i in ind.
template<typename T>
void indexed_divide(tensor_t<T> lhs, const tensor_t<T> rhs, const index_t ind);

//! lhs[i] = mix(lhs[i], rhs[i], w), for i in ind.
template<typename T>
void indexed_mix(tensor_t<T> lhs, const tensor_t<T> rhs, const index_t ind, const float w);

//
// Indexed Reductions
//

//! Sum of lhs[i], for i in ind.
template<typename T>
T indexed_sum(const tensor_t<T> lhs, const index_t ind);

//! Mean of lhs[i], for i in ind.
template<typename T>
T indexed_mean(const tensor_t<T> lhs, const index_t ind);

//! Min of lhs[i], for i in ind. NaN-skipping, like the dense min().
template<typename T>
T indexed_min(const tensor_t<T> lhs, const index_t ind);

//! Max of lhs[i], for i in ind. NaN-skipping, like the dense max().
template<typename T>
T indexed_max(const tensor_t<T> lhs, const index_t ind);

//! Flat index into lhs of the minimal element among lhs[i], i in ind.
template<typename T>
int64_t indexed_argmin(const tensor_t<T> lhs, const index_t ind);

//! Flat index into lhs of the maximal element among lhs[i], i in ind.
template<typename T>
int64_t indexed_argmax(const tensor_t<T> lhs, const index_t ind);

} // namespace silt
