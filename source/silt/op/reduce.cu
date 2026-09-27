#include <silt/core/error.hpp>
#include <silt/core/operation.hpp>
#include <silt/op/reduce.hpp>

#include <cub/device/device_reduce.cuh>
#include <thrust/execution_policy.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform.h>

#include <cmath>
#include <limits>

namespace silt {

namespace detail {

//
// CPU Loops
//

template<typename T>
T sum_cpu(const tensor_t<T>& tensor) {
  T acc = T(0);
  for (size_t i = 0; i < tensor.elem(); ++i)
    acc += tensor[i];
  return acc;
}

template<typename T>
T sumsq_cpu(const tensor_t<T>& tensor) {
  T acc = T(0);
  for (size_t i = 0; i < tensor.elem(); ++i) {
    const T v = tensor[i];
    acc += v * v;
  }
  return acc;
}

template<typename T>
T min_cpu(const tensor_t<T>& tensor) {
  T val = std::numeric_limits<T>::max();
  for (size_t i = 0; i < tensor.elem(); ++i) {
    const T v = tensor[i];
    if constexpr (std::is_floating_point_v<T>) {
      if (std::isnan(v)) continue;
    }
    val = std::min(val, v);
  }
  return val;
}

template<typename T>
T max_cpu(const tensor_t<T>& tensor) {
  T val = std::numeric_limits<T>::lowest();
  for (size_t i = 0; i < tensor.elem(); ++i) {
    const T v = tensor[i];
    if constexpr (std::is_floating_point_v<T>) {
      if (std::isnan(v)) continue;
    }
    val = std::max(val, v);
  }
  return val;
}

template<typename T>
int64_t argmin_cpu(const tensor_t<T>& tensor) {
  int64_t best = 0;
  T best_val = tensor[0];
  for (size_t i = 1; i < tensor.elem(); ++i) {
    const T v = tensor[i];
    if (v < best_val) {
      best_val = v;
      best = (int64_t)i;
    }
  }
  return best;
}

template<typename T>
int64_t argmax_cpu(const tensor_t<T>& tensor) {
  int64_t best = 0;
  T best_val = tensor[0];
  for (size_t i = 1; i < tensor.elem(); ++i) {
    const T v = tensor[i];
    if (v > best_val) {
      best_val = v;
      best = (int64_t)i;
    }
  }
  return best;
}

//
// GPU: cub::DeviceReduce, run twice (sizing pass, then real pass).
//

template<typename O, typename F>
O reduce_to_host(F pass) {
  O* d_out = (O*)silt::device_alloc(sizeof(O));

  void* d_temp = nullptr;
  size_t temp_bytes = 0;
  gpuErrchk(pass(d_temp, temp_bytes, d_out));
  gpuErrchk(cudaMalloc(&d_temp, temp_bytes));
  gpuErrchk(pass(d_temp, temp_bytes, d_out));
  gpuErrchk(cudaGetLastError());

  O result;
  gpuErrchk(cudaMemcpy(&result, d_out, sizeof(O), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaFree(d_temp));
  gpuErrchk(cudaFree(d_out));
  return result;
}

template<typename T>
struct square_op {
  GPU_ENABLE T operator()(const T x) const { return x * x; }
};

//! Maps NaN to +infinity (numeric_limits::max() for int, which has none) so
//! a plain cub::DeviceReduce::Min never picks it. Preprocessing, rather
//! than a custom reduction op, since cub's convenience wrappers are the
//! well-exercised path.
template<typename T>
struct nan_to_max_op {
  GPU_ENABLE T operator()(const T x) const {
    if constexpr (std::is_floating_point_v<T>) {
      if (isnan(x)) return std::numeric_limits<T>::max();
    }
    return x;
  }
};

//! Maps NaN to -infinity (numeric_limits::lowest()), the Max counterpart
//! of nan_to_max_op.
template<typename T>
struct nan_to_lowest_op {
  GPU_ENABLE T operator()(const T x) const {
    if constexpr (std::is_floating_point_v<T>) {
      if (isnan(x)) return std::numeric_limits<T>::lowest();
    }
    return x;
  }
};

template<typename T>
T sum_gpu(const tensor_t<T>& tensor) {
  const int64_t n = (int64_t)tensor.elem();
  const T* d_in = tensor.data();
  return reduce_to_host<T>([d_in, n](void* d_temp, size_t& temp_bytes, T* d_out) {
    return cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, n);
  });
}

template<typename T>
T sumsq_gpu(const tensor_t<T>& tensor) {
  const int64_t n = (int64_t)tensor.elem();
  thrust::transform_iterator<square_op<T>, const T*> sq(tensor.data(), square_op<T>{});
  return reduce_to_host<T>([sq, n](void* d_temp, size_t& temp_bytes, T* d_out) {
    return cub::DeviceReduce::Sum(d_temp, temp_bytes, sq, d_out, n);
  });
}

// NaN-mapped values are materialized into a scratch buffer first, then
// reduced with cub over a plain pointer -- matches sum_gpu's proven
// raw-pointer path rather than feeding cub a nested iterator adaptor.
template<typename T>
T min_gpu(const tensor_t<T>& tensor) {
  const int64_t n = (int64_t)tensor.elem();
  T* scratch = (T*)silt::device_alloc(sizeof(T) * (size_t)n);
  thrust::transform(thrust::device, tensor.data(), tensor.data() + n, scratch, nan_to_max_op<T>{});
  const T result = reduce_to_host<T>([scratch, n](void* d_temp, size_t& temp_bytes, T* d_out) {
    return cub::DeviceReduce::Min(d_temp, temp_bytes, scratch, d_out, n);
  });
  gpuErrchk(cudaFree(scratch));
  return result;
}

template<typename T>
T max_gpu(const tensor_t<T>& tensor) {
  const int64_t n = (int64_t)tensor.elem();
  T* scratch = (T*)silt::device_alloc(sizeof(T) * (size_t)n);
  thrust::transform(thrust::device, tensor.data(), tensor.data() + n, scratch, nan_to_lowest_op<T>{});
  const T result = reduce_to_host<T>([scratch, n](void* d_temp, size_t& temp_bytes, T* d_out) {
    return cub::DeviceReduce::Max(d_temp, temp_bytes, scratch, d_out, n);
  });
  gpuErrchk(cudaFree(scratch));
  return result;
}

template<typename T>
int64_t argmin_gpu(const tensor_t<T>& tensor) {
  const int n = (int)tensor.elem(); // cub::DeviceReduce::ArgMin's num_items is `int`
  const T* d_in = tensor.data();
  using KV = cub::KeyValuePair<int, T>;
  const KV result = reduce_to_host<KV>([d_in, n](void* d_temp, size_t& temp_bytes, KV* d_out) {
    return cub::DeviceReduce::ArgMin(d_temp, temp_bytes, d_in, d_out, n);
  });
  return (int64_t)result.key;
}

template<typename T>
int64_t argmax_gpu(const tensor_t<T>& tensor) {
  const int n = (int)tensor.elem();
  const T* d_in = tensor.data();
  using KV = cub::KeyValuePair<int, T>;
  const KV result = reduce_to_host<KV>([d_in, n](void* d_temp, size_t& temp_bytes, KV* d_out) {
    return cub::DeviceReduce::ArgMax(d_temp, temp_bytes, d_in, d_out, n);
  });
  return (int64_t)result.key;
}

} // namespace detail

//
// Public Dispatch (CPU / GPU)
//

template<typename T>
T sum(const tensor_t<T>& tensor) {
  if (tensor.host() == silt::host_t::CPU)
    return detail::sum_cpu(tensor);
  return detail::sum_gpu(tensor);
}

template<typename T>
T mean(const tensor_t<T>& tensor) {
  return sum(tensor) / (T)tensor.elem();
}

template<typename T>
T min(const tensor_t<T>& tensor) {
  if (tensor.host() == silt::host_t::CPU)
    return detail::min_cpu(tensor);
  return detail::min_gpu(tensor);
}

template<typename T>
T max(const tensor_t<T>& tensor) {
  if (tensor.host() == silt::host_t::CPU)
    return detail::max_cpu(tensor);
  return detail::max_gpu(tensor);
}

template<typename T>
T variance(const tensor_t<T>& tensor) {
  const T n = (T)tensor.elem();
  const T m = mean(tensor);
  const T sumsq = (tensor.host() == silt::host_t::CPU)
                      ? detail::sumsq_cpu(tensor)
                      : detail::sumsq_gpu(tensor);
  return sumsq / n - m * m;
}

template<typename T>
T stddev(const tensor_t<T>& tensor) {
  return (T)std::sqrt((double)variance(tensor));
}

template<typename T>
int64_t argmin(const tensor_t<T>& tensor) {
  if (tensor.host() == silt::host_t::CPU)
    return detail::argmin_cpu(tensor);
  return detail::argmin_gpu(tensor);
}

template<typename T>
int64_t argmax(const tensor_t<T>& tensor) {
  if (tensor.host() == silt::host_t::CPU)
    return detail::argmax_cpu(tensor);
  return detail::argmax_gpu(tensor);
}

template EXPORT_SHARED int silt::sum<int>(const silt::tensor_t<int>& tensor);
template EXPORT_SHARED float silt::sum<float>(const silt::tensor_t<float>& tensor);
template EXPORT_SHARED double silt::sum<double>(const silt::tensor_t<double>& tensor);

template EXPORT_SHARED int silt::mean<int>(const silt::tensor_t<int>& tensor);
template EXPORT_SHARED float silt::mean<float>(const silt::tensor_t<float>& tensor);
template EXPORT_SHARED double silt::mean<double>(const silt::tensor_t<double>& tensor);

template EXPORT_SHARED int silt::min<int>(const silt::tensor_t<int>& tensor);
template EXPORT_SHARED float silt::min<float>(const silt::tensor_t<float>& tensor);
template EXPORT_SHARED double silt::min<double>(const silt::tensor_t<double>& tensor);

template EXPORT_SHARED int silt::max<int>(const silt::tensor_t<int>& tensor);
template EXPORT_SHARED float silt::max<float>(const silt::tensor_t<float>& tensor);
template EXPORT_SHARED double silt::max<double>(const silt::tensor_t<double>& tensor);

template EXPORT_SHARED int silt::variance<int>(const silt::tensor_t<int>& tensor);
template EXPORT_SHARED float silt::variance<float>(const silt::tensor_t<float>& tensor);
template EXPORT_SHARED double silt::variance<double>(const silt::tensor_t<double>& tensor);

template EXPORT_SHARED int silt::stddev<int>(const silt::tensor_t<int>& tensor);
template EXPORT_SHARED float silt::stddev<float>(const silt::tensor_t<float>& tensor);
template EXPORT_SHARED double silt::stddev<double>(const silt::tensor_t<double>& tensor);

template EXPORT_SHARED int64_t silt::argmin<int>(const silt::tensor_t<int>& tensor);
template EXPORT_SHARED int64_t silt::argmin<float>(const silt::tensor_t<float>& tensor);
template EXPORT_SHARED int64_t silt::argmin<double>(const silt::tensor_t<double>& tensor);

template EXPORT_SHARED int64_t silt::argmax<int>(const silt::tensor_t<int>& tensor);
template EXPORT_SHARED int64_t silt::argmax<float>(const silt::tensor_t<float>& tensor);
template EXPORT_SHARED int64_t silt::argmax<double>(const silt::tensor_t<double>& tensor);

} // end of namespace silt
