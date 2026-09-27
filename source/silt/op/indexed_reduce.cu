#include <silt/core/error.hpp>
#include <silt/core/operation.hpp>
#include <silt/op/indexed.hpp>

#include <cub/device/device_reduce.cuh>
#include <thrust/execution_policy.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform.h>

#include <cmath>
#include <limits>

namespace silt {

namespace detail {

//
// CPU Loops -- gather lhs[ind[k]] directly, no scratch buffer.
//

template<typename T>
T indexed_sum_cpu(const tensor_t<T> lhs, const index_t ind) {
  T acc = T(0);
  for (int64_t k = 0; k < (int64_t)ind.elem(); ++k)
    acc += lhs[ind[k]];
  return acc;
}

template<typename T>
T indexed_min_cpu(const tensor_t<T> lhs, const index_t ind) {
  T val = std::numeric_limits<T>::max();
  for (int64_t k = 0; k < (int64_t)ind.elem(); ++k) {
    const T v = lhs[ind[k]];
    if constexpr (std::is_floating_point_v<T>) {
      if (std::isnan(v)) continue;
    }
    val = std::min(val, v);
  }
  return val;
}

template<typename T>
T indexed_max_cpu(const tensor_t<T> lhs, const index_t ind) {
  T val = std::numeric_limits<T>::lowest();
  for (int64_t k = 0; k < (int64_t)ind.elem(); ++k) {
    const T v = lhs[ind[k]];
    if constexpr (std::is_floating_point_v<T>) {
      if (std::isnan(v)) continue;
    }
    val = std::max(val, v);
  }
  return val;
}

template<typename T>
int64_t indexed_argmin_cpu(const tensor_t<T> lhs, const index_t ind) {
  int64_t best = ind[0];
  T best_val = lhs[best];
  for (int64_t k = 1; k < (int64_t)ind.elem(); ++k) {
    const int64_t i = ind[k];
    const T v = lhs[i];
    if (v < best_val) {
      best_val = v;
      best = i;
    }
  }
  return best;
}

template<typename T>
int64_t indexed_argmax_cpu(const tensor_t<T> lhs, const index_t ind) {
  int64_t best = ind[0];
  T best_val = lhs[best];
  for (int64_t k = 1; k < (int64_t)ind.elem(); ++k) {
    const int64_t i = ind[k];
    const T v = lhs[i];
    if (v > best_val) {
      best_val = v;
      best = i;
    }
  }
  return best;
}

//
// GPU: gather through `ind` with a permutation iterator, reduce with cub.
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
using gather_t = thrust::permutation_iterator<const T*, const int64_t*>;

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
T indexed_sum_gpu(const tensor_t<T> lhs, const index_t ind) {
  const int64_t n = (int64_t)ind.elem();
  const gather_t<T> gather(lhs.data(), ind.data());
  return reduce_to_host<T>([gather, n](void* d_temp, size_t& temp_bytes, T* d_out) {
    return cub::DeviceReduce::Sum(d_temp, temp_bytes, gather, d_out, n);
  });
}

// Gathered + NaN-mapped values are materialized into a scratch buffer
// first, then reduced with cub over a plain pointer -- matches
// indexed_sum_gpu's proven pattern of handing cub a single, un-nested
// iterator rather than a transform_iterator wrapped around a
// permutation_iterator.
template<typename T>
T indexed_min_gpu(const tensor_t<T> lhs, const index_t ind) {
  const int64_t n = (int64_t)ind.elem();
  const gather_t<T> gather(lhs.data(), ind.data());
  T* scratch = (T*)silt::device_alloc(sizeof(T) * (size_t)n);
  thrust::transform(thrust::device, gather, gather + n, scratch, nan_to_max_op<T>{});
  const T result = reduce_to_host<T>([scratch, n](void* d_temp, size_t& temp_bytes, T* d_out) {
    return cub::DeviceReduce::Min(d_temp, temp_bytes, scratch, d_out, n);
  });
  gpuErrchk(cudaFree(scratch));
  return result;
}

template<typename T>
T indexed_max_gpu(const tensor_t<T> lhs, const index_t ind) {
  const int64_t n = (int64_t)ind.elem();
  const gather_t<T> gather(lhs.data(), ind.data());
  T* scratch = (T*)silt::device_alloc(sizeof(T) * (size_t)n);
  thrust::transform(thrust::device, gather, gather + n, scratch, nan_to_lowest_op<T>{});
  const T result = reduce_to_host<T>([scratch, n](void* d_temp, size_t& temp_bytes, T* d_out) {
    return cub::DeviceReduce::Max(d_temp, temp_bytes, scratch, d_out, n);
  });
  gpuErrchk(cudaFree(scratch));
  return result;
}

//! ind[key] -- single-element device-to-host lookup.
inline int64_t index_at(const index_t ind, const int64_t key) {
  int64_t value;
  gpuErrchk(cudaMemcpy(&value, ind.data() + key, sizeof(int64_t), cudaMemcpyDeviceToHost));
  return value;
}

template<typename T>
int64_t indexed_argmin_gpu(const tensor_t<T> lhs, const index_t ind) {
  const int n = (int)ind.elem(); // cub::DeviceReduce::ArgMin's num_items is `int`
  const gather_t<T> gather(lhs.data(), ind.data());
  using KV = cub::KeyValuePair<int, T>;
  const KV result = reduce_to_host<KV>([gather, n](void* d_temp, size_t& temp_bytes, KV* d_out) {
    return cub::DeviceReduce::ArgMin(d_temp, temp_bytes, gather, d_out, n);
  });
  // result.key indexes into `ind`, not `lhs` -- look up the underlying flat index.
  return index_at(ind, (int64_t)result.key);
}

template<typename T>
int64_t indexed_argmax_gpu(const tensor_t<T> lhs, const index_t ind) {
  const int n = (int)ind.elem();
  const gather_t<T> gather(lhs.data(), ind.data());
  using KV = cub::KeyValuePair<int, T>;
  const KV result = reduce_to_host<KV>([gather, n](void* d_temp, size_t& temp_bytes, KV* d_out) {
    return cub::DeviceReduce::ArgMax(d_temp, temp_bytes, gather, d_out, n);
  });
  return index_at(ind, (int64_t)result.key);
}

} // namespace detail

//
// Public Dispatch (CPU / GPU)
//

template<typename T>
T indexed_sum(const tensor_t<T> lhs, const index_t ind) {
  if (lhs.host() != ind.host())
    throw silt::error::mismatch_host(lhs.host(), ind.host());
  if (lhs.host() == silt::host_t::CPU)
    return detail::indexed_sum_cpu(lhs, ind);
  return detail::indexed_sum_gpu(lhs, ind);
}

template<typename T>
T indexed_mean(const tensor_t<T> lhs, const index_t ind) {
  return indexed_sum<T>(lhs, ind) / (T)ind.elem();
}

template<typename T>
T indexed_min(const tensor_t<T> lhs, const index_t ind) {
  if (lhs.host() != ind.host())
    throw silt::error::mismatch_host(lhs.host(), ind.host());
  if (lhs.host() == silt::host_t::CPU)
    return detail::indexed_min_cpu(lhs, ind);
  return detail::indexed_min_gpu(lhs, ind);
}

template<typename T>
T indexed_max(const tensor_t<T> lhs, const index_t ind) {
  if (lhs.host() != ind.host())
    throw silt::error::mismatch_host(lhs.host(), ind.host());
  if (lhs.host() == silt::host_t::CPU)
    return detail::indexed_max_cpu(lhs, ind);
  return detail::indexed_max_gpu(lhs, ind);
}

template<typename T>
int64_t indexed_argmin(const tensor_t<T> lhs, const index_t ind) {
  if (lhs.host() != ind.host())
    throw silt::error::mismatch_host(lhs.host(), ind.host());
  if (lhs.host() == silt::host_t::CPU)
    return detail::indexed_argmin_cpu(lhs, ind);
  return detail::indexed_argmin_gpu(lhs, ind);
}

template<typename T>
int64_t indexed_argmax(const tensor_t<T> lhs, const index_t ind) {
  if (lhs.host() != ind.host())
    throw silt::error::mismatch_host(lhs.host(), ind.host());
  if (lhs.host() == silt::host_t::CPU)
    return detail::indexed_argmax_cpu(lhs, ind);
  return detail::indexed_argmax_gpu(lhs, ind);
}

template EXPORT_SHARED int silt::indexed_sum<int>(silt::tensor_t<int> lhs, const index_t ind);
template EXPORT_SHARED float silt::indexed_sum<float>(silt::tensor_t<float> lhs, const index_t ind);
template EXPORT_SHARED double silt::indexed_sum<double>(silt::tensor_t<double> lhs, const index_t ind);

template EXPORT_SHARED int silt::indexed_mean<int>(silt::tensor_t<int> lhs, const index_t ind);
template EXPORT_SHARED float silt::indexed_mean<float>(silt::tensor_t<float> lhs, const index_t ind);
template EXPORT_SHARED double silt::indexed_mean<double>(silt::tensor_t<double> lhs, const index_t ind);

template EXPORT_SHARED int silt::indexed_min<int>(silt::tensor_t<int> lhs, const index_t ind);
template EXPORT_SHARED float silt::indexed_min<float>(silt::tensor_t<float> lhs, const index_t ind);
template EXPORT_SHARED double silt::indexed_min<double>(silt::tensor_t<double> lhs, const index_t ind);

template EXPORT_SHARED int silt::indexed_max<int>(silt::tensor_t<int> lhs, const index_t ind);
template EXPORT_SHARED float silt::indexed_max<float>(silt::tensor_t<float> lhs, const index_t ind);
template EXPORT_SHARED double silt::indexed_max<double>(silt::tensor_t<double> lhs, const index_t ind);

template EXPORT_SHARED int64_t silt::indexed_argmin<int>(silt::tensor_t<int> lhs, const index_t ind);
template EXPORT_SHARED int64_t silt::indexed_argmin<float>(silt::tensor_t<float> lhs, const index_t ind);
template EXPORT_SHARED int64_t silt::indexed_argmin<double>(silt::tensor_t<double> lhs, const index_t ind);

template EXPORT_SHARED int64_t silt::indexed_argmax<int>(silt::tensor_t<int> lhs, const index_t ind);
template EXPORT_SHARED int64_t silt::indexed_argmax<float>(silt::tensor_t<float> lhs, const index_t ind);
template EXPORT_SHARED int64_t silt::indexed_argmax<double>(silt::tensor_t<double> lhs, const index_t ind);

} // end of namespace silt
