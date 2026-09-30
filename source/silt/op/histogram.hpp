#pragma once

#include <silt/core/tensor.hpp>
#include <silt/core/view.hpp>

namespace silt {

//
// Histogram (CPU + GPU)
//

//! Counts of the elements of `data` in `bins` equal-width bins over [lo, hi].
//!
//! Bins are half-open except the last, which includes `hi`, so the range
//! [lo, hi] is fully covered. Elements outside the range, and NaNs, are not
//! counted. Throws std::invalid_argument unless bins >= 1 and lo < hi are
//! finite.
//!
//! The result is a 1D tensor of `bins` counts on the host of `data`. To
//! restrict to an index set, gather first.
template<typename T>
tensor_t<int> histogram(const tensor_t<T> data, const int bins, const T lo, const T hi);

//! Histogram of the logical (slice-space) elements of a view.
template<typename T>
tensor_t<int> histogram(const view_t<T> data, const int bins, const T lo, const T hi);

} // namespace silt
