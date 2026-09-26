#pragma once

#include <silt/core/shape.hpp>
#include <silt/core/tensor.hpp>
#include <silt/core/types.hpp>

namespace silt {

//
// Indexed Operations Definitions
//

template<typename T>
void indexed_set(tensor_t<T> lhs, const T rhs, const tensor_t<int> ind);

//
// Index Generation Function:
//  Here, based on distance from some center.
//

EXPORT_SHARED tensor_t<int> index_radius(const silt::shape shape, const silt::vec2 center, const float rad);

} // namespace silt
