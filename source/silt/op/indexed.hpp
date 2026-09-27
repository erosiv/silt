#pragma once

#include <silt/silt.hpp>
#include <silt/core/shape.hpp>
#include <silt/core/tensor.hpp>
#include <silt/core/types.hpp>

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

} // namespace silt
