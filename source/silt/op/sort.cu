#include <silt/op/sort.hpp>

#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cmath>
#include <type_traits>

namespace silt {

namespace detail {

//! Strict weak ordering with NaN as the greatest value; std::sort is
//! undefined for `<` alone when NaNs are present.
template<typename T>
struct nan_last_less {
  bool operator()(const T a, const T b) const {
    if constexpr (std::is_floating_point_v<T>)
      return a < b || (std::isnan(b) && !std::isnan(a));
    else
      return a < b;
  }
};

} // namespace detail

template<typename T>
void sort(tensor_t<T> tensor) {

  if (tensor.elem() < 2)
    return;

  T* first = tensor.data();
  T* last = first + tensor.elem();

  if (tensor.host() == silt::host_t::CPU)
    std::sort(first, last, detail::nan_last_less<T>{});
  else
    thrust::sort(thrust::device, first, last);
}

template EXPORT_SHARED void silt::sort<int>(silt::tensor_t<int> tensor);
template EXPORT_SHARED void silt::sort<float>(silt::tensor_t<float> tensor);
template EXPORT_SHARED void silt::sort<double>(silt::tensor_t<double> tensor);

} // end of namespace silt
