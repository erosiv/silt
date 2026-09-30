#pragma once

#include <silt/core/tensor.hpp>
#include <silt/op/indexed.hpp>

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

//! Permutation that sorts `tensor` ascending: out[k] is the flat index of the
//! k-th smallest element. Stable -- equal elements keep their flat order.
//! NaN placement is as for sort(). The result is an int64 tensor on the host
//! of `tensor`; it is not a sorted index set (see index_sort_unique).
template<typename T>
index_t argsort(const tensor_t<T> tensor);

} // namespace silt
