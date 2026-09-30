#pragma once

#include <silt/core/tensor.hpp>

namespace silt {

//
// Dense Sorting (whole-buffer, CPU + GPU)
//

//! In-place ascending sort of every element, in flat order.
//!
//! NaN placement depends on the host: the CPU sorts NaNs last, while the GPU
//! radix sort orders them by sign bit (positive NaNs last, negative first).
//! Remove NaNs beforehand if the position matters.
template<typename T>
void sort(tensor_t<T> tensor);

} // namespace silt
