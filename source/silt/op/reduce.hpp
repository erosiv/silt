#pragma once

#include <silt/core/tensor.hpp>

namespace silt {

//
// Dense Reductions (whole-buffer, CPU + GPU)
//

template<typename T>
T sum(const tensor_t<T>& tensor);

template<typename T>
T mean(const tensor_t<T>& tensor);

//! NaN-skipping (a NaN element is ignored rather than propagated).
template<typename T>
T min(const tensor_t<T>& tensor);

//! NaN-skipping (a NaN element is ignored rather than propagated).
template<typename T>
T max(const tensor_t<T>& tensor);

//! Population variance (ddof=0).
template<typename T>
T variance(const tensor_t<T>& tensor);

template<typename T>
T stddev(const tensor_t<T>& tensor);

//! Flat index of the first minimal element.
template<typename T>
int64_t argmin(const tensor_t<T>& tensor);

//! Flat index of the first maximal element.
template<typename T>
int64_t argmax(const tensor_t<T>& tensor);

} // namespace silt
