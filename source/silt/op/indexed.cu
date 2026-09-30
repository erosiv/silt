#include <silt/core/operation.hpp>
#include <silt/op/indexed.hpp>

#include <cub/device/device_select.cuh>
#include <thrust/iterator/counting_iterator.h>

namespace silt {

//
// Indexed Assignment / Arithmetic
//

template<typename T>
void indexed_set(tensor_t<T> lhs, const T rhs, const index_t ind) {
  op::indexed_apply(lhs, ind, [rhs] GPU_ENABLE(const T a) {
    return rhs;
  });
}

template<typename T>
void indexed_add(tensor_t<T> lhs, const T rhs, const index_t ind) {
  op::indexed_apply(lhs, ind, [rhs] GPU_ENABLE(const T a) {
    return a + rhs;
  });
}

template<typename T>
void indexed_add(tensor_t<T> lhs, const tensor_t<T> rhs, const index_t ind) {
  op::indexed_binop_apply(lhs, rhs, ind, [] GPU_ENABLE(const T a, const T b) {
    return a + b;
  });
}

template<typename T>
void indexed_multiply(tensor_t<T> lhs, const T rhs, const index_t ind) {
  op::indexed_apply(lhs, ind, [rhs] GPU_ENABLE(const T a) {
    return a * rhs;
  });
}

template<typename T>
void indexed_multiply(tensor_t<T> lhs, const tensor_t<T> rhs, const index_t ind) {
  op::indexed_binop_apply(lhs, rhs, ind, [] GPU_ENABLE(const T a, const T b) {
    return a * b;
  });
}

template<typename T>
void indexed_divide(tensor_t<T> lhs, const T rhs, const index_t ind) {
  op::indexed_apply(lhs, ind, [rhs] GPU_ENABLE(const T a) {
    return a / rhs;
  });
}

template<typename T>
void indexed_divide(tensor_t<T> lhs, const tensor_t<T> rhs, const index_t ind) {
  op::indexed_binop_apply(lhs, rhs, ind, [] GPU_ENABLE(const T a, const T b) {
    return a / b;
  });
}

template<typename T>
void indexed_mix(tensor_t<T> lhs, const tensor_t<T> rhs, const index_t ind, const float w) {
  op::indexed_binop_apply(lhs, rhs, ind, [w] GPU_ENABLE(const T a, const T b) {
    return (1.0f - w) * a + w * b;
  });
}

//
// Indexed Assignment / Arithmetic (View)
//
//  Same operations in the slice-space of a view.
//

template<typename T>
void indexed_set(view_t<T> lhs, const T rhs, const index_t ind) {
  op::indexed_apply(lhs, ind, [rhs] GPU_ENABLE(const T a) {
    return rhs;
  });
}

template<typename T>
void indexed_add(view_t<T> lhs, const T rhs, const index_t ind) {
  op::indexed_apply(lhs, ind, [rhs] GPU_ENABLE(const T a) {
    return a + rhs;
  });
}

template<typename T>
void indexed_add(view_t<T> lhs, const view_t<T> rhs, const index_t ind) {
  op::indexed_binop_apply(lhs, rhs, ind, [] GPU_ENABLE(const T a, const T b) {
    return a + b;
  });
}

template<typename T>
void indexed_multiply(view_t<T> lhs, const T rhs, const index_t ind) {
  op::indexed_apply(lhs, ind, [rhs] GPU_ENABLE(const T a) {
    return a * rhs;
  });
}

template<typename T>
void indexed_multiply(view_t<T> lhs, const view_t<T> rhs, const index_t ind) {
  op::indexed_binop_apply(lhs, rhs, ind, [] GPU_ENABLE(const T a, const T b) {
    return a * b;
  });
}

template<typename T>
void indexed_divide(view_t<T> lhs, const T rhs, const index_t ind) {
  op::indexed_apply(lhs, ind, [rhs] GPU_ENABLE(const T a) {
    return a / rhs;
  });
}

template<typename T>
void indexed_divide(view_t<T> lhs, const view_t<T> rhs, const index_t ind) {
  op::indexed_binop_apply(lhs, rhs, ind, [] GPU_ENABLE(const T a, const T b) {
    return a / b;
  });
}

template<typename T>
void indexed_mix(view_t<T> lhs, const view_t<T> rhs, const index_t ind, const float w) {
  op::indexed_binop_apply(lhs, rhs, ind, [w] GPU_ENABLE(const T a, const T b) {
    return (1.0f - w) * a + w * b;
  });
}

template EXPORT_SHARED void silt::indexed_set<int>(silt::tensor_t<int> lhs, const int rhs, const index_t ind);
template EXPORT_SHARED void silt::indexed_set<float>(silt::tensor_t<float> lhs, const float rhs, const index_t ind);
template EXPORT_SHARED void silt::indexed_set<double>(silt::tensor_t<double> lhs, const double rhs, const index_t ind);

template EXPORT_SHARED void silt::indexed_add<int>(silt::tensor_t<int> lhs, const int rhs, const index_t ind);
template EXPORT_SHARED void silt::indexed_add<float>(silt::tensor_t<float> lhs, const float rhs, const index_t ind);
template EXPORT_SHARED void silt::indexed_add<double>(silt::tensor_t<double> lhs, const double rhs, const index_t ind);

template EXPORT_SHARED void silt::indexed_add<int>(silt::tensor_t<int> lhs, const silt::tensor_t<int> rhs, const index_t ind);
template EXPORT_SHARED void silt::indexed_add<float>(silt::tensor_t<float> lhs, const silt::tensor_t<float> rhs, const index_t ind);
template EXPORT_SHARED void silt::indexed_add<double>(silt::tensor_t<double> lhs, const silt::tensor_t<double> rhs, const index_t ind);

template EXPORT_SHARED void silt::indexed_multiply<int>(silt::tensor_t<int> lhs, const int rhs, const index_t ind);
template EXPORT_SHARED void silt::indexed_multiply<float>(silt::tensor_t<float> lhs, const float rhs, const index_t ind);
template EXPORT_SHARED void silt::indexed_multiply<double>(silt::tensor_t<double> lhs, const double rhs, const index_t ind);

template EXPORT_SHARED void silt::indexed_multiply<int>(silt::tensor_t<int> lhs, const silt::tensor_t<int> rhs, const index_t ind);
template EXPORT_SHARED void silt::indexed_multiply<float>(silt::tensor_t<float> lhs, const silt::tensor_t<float> rhs, const index_t ind);
template EXPORT_SHARED void silt::indexed_multiply<double>(silt::tensor_t<double> lhs, const silt::tensor_t<double> rhs, const index_t ind);

template EXPORT_SHARED void silt::indexed_divide<int>(silt::tensor_t<int> lhs, const int rhs, const index_t ind);
template EXPORT_SHARED void silt::indexed_divide<float>(silt::tensor_t<float> lhs, const float rhs, const index_t ind);
template EXPORT_SHARED void silt::indexed_divide<double>(silt::tensor_t<double> lhs, const double rhs, const index_t ind);

template EXPORT_SHARED void silt::indexed_divide<int>(silt::tensor_t<int> lhs, const silt::tensor_t<int> rhs, const index_t ind);
template EXPORT_SHARED void silt::indexed_divide<float>(silt::tensor_t<float> lhs, const silt::tensor_t<float> rhs, const index_t ind);
template EXPORT_SHARED void silt::indexed_divide<double>(silt::tensor_t<double> lhs, const silt::tensor_t<double> rhs, const index_t ind);

template EXPORT_SHARED void silt::indexed_mix<int>(silt::tensor_t<int> lhs, const silt::tensor_t<int> rhs, const index_t ind, const float w);
template EXPORT_SHARED void silt::indexed_mix<float>(silt::tensor_t<float> lhs, const silt::tensor_t<float> rhs, const index_t ind, const float w);
template EXPORT_SHARED void silt::indexed_mix<double>(silt::tensor_t<double> lhs, const silt::tensor_t<double> rhs, const index_t ind, const float w);

// View Overloads

template EXPORT_SHARED void silt::indexed_set<int>(silt::view_t<int> lhs, const int rhs, const index_t ind);
template EXPORT_SHARED void silt::indexed_set<float>(silt::view_t<float> lhs, const float rhs, const index_t ind);
template EXPORT_SHARED void silt::indexed_set<double>(silt::view_t<double> lhs, const double rhs, const index_t ind);

template EXPORT_SHARED void silt::indexed_add<int>(silt::view_t<int> lhs, const int rhs, const index_t ind);
template EXPORT_SHARED void silt::indexed_add<float>(silt::view_t<float> lhs, const float rhs, const index_t ind);
template EXPORT_SHARED void silt::indexed_add<double>(silt::view_t<double> lhs, const double rhs, const index_t ind);

template EXPORT_SHARED void silt::indexed_add<int>(silt::view_t<int> lhs, const silt::view_t<int> rhs, const index_t ind);
template EXPORT_SHARED void silt::indexed_add<float>(silt::view_t<float> lhs, const silt::view_t<float> rhs, const index_t ind);
template EXPORT_SHARED void silt::indexed_add<double>(silt::view_t<double> lhs, const silt::view_t<double> rhs, const index_t ind);

template EXPORT_SHARED void silt::indexed_multiply<int>(silt::view_t<int> lhs, const int rhs, const index_t ind);
template EXPORT_SHARED void silt::indexed_multiply<float>(silt::view_t<float> lhs, const float rhs, const index_t ind);
template EXPORT_SHARED void silt::indexed_multiply<double>(silt::view_t<double> lhs, const double rhs, const index_t ind);

template EXPORT_SHARED void silt::indexed_multiply<int>(silt::view_t<int> lhs, const silt::view_t<int> rhs, const index_t ind);
template EXPORT_SHARED void silt::indexed_multiply<float>(silt::view_t<float> lhs, const silt::view_t<float> rhs, const index_t ind);
template EXPORT_SHARED void silt::indexed_multiply<double>(silt::view_t<double> lhs, const silt::view_t<double> rhs, const index_t ind);

template EXPORT_SHARED void silt::indexed_divide<int>(silt::view_t<int> lhs, const int rhs, const index_t ind);
template EXPORT_SHARED void silt::indexed_divide<float>(silt::view_t<float> lhs, const float rhs, const index_t ind);
template EXPORT_SHARED void silt::indexed_divide<double>(silt::view_t<double> lhs, const double rhs, const index_t ind);

template EXPORT_SHARED void silt::indexed_divide<int>(silt::view_t<int> lhs, const silt::view_t<int> rhs, const index_t ind);
template EXPORT_SHARED void silt::indexed_divide<float>(silt::view_t<float> lhs, const silt::view_t<float> rhs, const index_t ind);
template EXPORT_SHARED void silt::indexed_divide<double>(silt::view_t<double> lhs, const silt::view_t<double> rhs, const index_t ind);

template EXPORT_SHARED void silt::indexed_mix<int>(silt::view_t<int> lhs, const silt::view_t<int> rhs, const index_t ind, const float w);
template EXPORT_SHARED void silt::indexed_mix<float>(silt::view_t<float> lhs, const silt::view_t<float> rhs, const index_t ind, const float w);
template EXPORT_SHARED void silt::indexed_mix<double>(silt::view_t<double> lhs, const silt::view_t<double> rhs, const index_t ind, const float w);

//
// Generic Compaction: Predicate -> Index Set
//

// GPU-only for now; no CPU compaction path.
template<typename F>
index_t make_index_set(const silt::shape shape, F predicate) {

  const int64_t n = shape.elem();

  // Worst-case scratch: every element could satisfy the predicate.
  index_t scratch(silt::shape((int)n), silt::host_t::GPU);

  int64_t* d_count = nullptr;
  gpuErrchk(cudaMalloc(&d_count, sizeof(int64_t)));

  thrust::counting_iterator<int64_t> first(0);

  void* d_temp = nullptr;
  size_t temp_bytes = 0;

  // Sizing pass.
  gpuErrchk(cub::DeviceSelect::If(d_temp, temp_bytes, first, scratch.data(), d_count, n, predicate));
  gpuErrchk(cudaMalloc(&d_temp, temp_bytes));
  // Actual compaction.
  gpuErrchk(cub::DeviceSelect::If(d_temp, temp_bytes, first, scratch.data(), d_count, n, predicate));
  gpuErrchk(cudaGetLastError());

  int64_t count = 0;
  gpuErrchk(cudaMemcpy(&count, d_count, sizeof(int64_t), cudaMemcpyDeviceToHost));

  gpuErrchk(cudaFree(d_count));
  gpuErrchk(cudaFree(d_temp));

  // Exact-sized result; count == 0 is legal.
  index_t result(silt::shape((int)count), silt::host_t::GPU);
  if (count > 0) {
    gpuErrchk(cudaMemcpy(result.data(), scratch.data(), count * sizeof(int64_t), cudaMemcpyDeviceToDevice));
  }
  return result;
}

template index_t silt::make_index_set<silt::radius_predicate>(const silt::shape shape, silt::radius_predicate predicate);
template index_t silt::make_index_set<silt::box_predicate>(const silt::shape shape, silt::box_predicate predicate);
template index_t silt::make_index_set<silt::polygon_predicate>(const silt::shape shape, silt::polygon_predicate predicate);
template index_t silt::make_index_set<silt::range_predicate<int>>(const silt::shape shape, silt::range_predicate<int> predicate);
template index_t silt::make_index_set<silt::range_predicate<float>>(const silt::shape shape, silt::range_predicate<float> predicate);
template index_t silt::make_index_set<silt::range_predicate<double>>(const silt::shape shape, silt::range_predicate<double> predicate);

//
// Selector Convenience Functions
//

index_t index_radius(const silt::shape shape, const silt::vec2 center, const float radius) {
  return make_index_set(shape, radius_predicate{shape, center, radius});
}

index_t index_box(const silt::shape shape, const silt::vec2 lo, const silt::vec2 hi) {
  return make_index_set(shape, box_predicate{shape, lo, hi});
}

index_t index_polygon(const silt::shape shape, const tensor_t<silt::vec2>& vertices) {
  if (vertices.host() != silt::host_t::GPU)
    throw silt::error::mismatch_host(silt::host_t::GPU, vertices.host());
  return make_index_set(shape, polygon_predicate{shape, vertices.data(), (int)vertices.elem()});
}

namespace detail {

__global__ void index_slice_kernel(index_t out, const silt::slice s) {
  const int64_t n = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (n < out.elem())
    out[n] = s.transform(n);
}

} // namespace detail

index_t index_slice(const silt::slice& s) {
  index_t result(silt::shape((int)s.elem()), silt::host_t::GPU);
  detail::index_slice_kernel<<<block(result.elem(), 512), 512>>>(result, s);
  gpuErrchk(cudaGetLastError());
  return result;
}

template<typename T>
index_t index_range(const tensor_t<T>& data, const T lo, const T hi) {
  if (data.host() != silt::host_t::GPU)
    throw silt::error::mismatch_host(silt::host_t::GPU, data.host());
  return make_index_set(data.shape(), range_predicate<T>{data.data(), lo, hi});
}

template<typename T>
index_t index_greater(const tensor_t<T>& data, const T value) {
  return index_range<T>(data, value, positive_infinity<T>());
}

template<typename T>
index_t index_lesser(const tensor_t<T>& data, const T value) {
  return index_range<T>(data, negative_infinity<T>(), value);
}

template<typename T>
index_t index_match(const tensor_t<T>& data, const T value) {
  return index_range<T>(data, value, value);
}

template EXPORT_SHARED index_t silt::index_range<int>(const silt::tensor_t<int>& data, const int lo, const int hi);
template EXPORT_SHARED index_t silt::index_range<float>(const silt::tensor_t<float>& data, const float lo, const float hi);
template EXPORT_SHARED index_t silt::index_range<double>(const silt::tensor_t<double>& data, const double lo, const double hi);

template EXPORT_SHARED index_t silt::index_greater<int>(const silt::tensor_t<int>& data, const int value);
template EXPORT_SHARED index_t silt::index_greater<float>(const silt::tensor_t<float>& data, const float value);
template EXPORT_SHARED index_t silt::index_greater<double>(const silt::tensor_t<double>& data, const double value);

template EXPORT_SHARED index_t silt::index_lesser<int>(const silt::tensor_t<int>& data, const int value);
template EXPORT_SHARED index_t silt::index_lesser<float>(const silt::tensor_t<float>& data, const float value);
template EXPORT_SHARED index_t silt::index_lesser<double>(const silt::tensor_t<double>& data, const double value);

template EXPORT_SHARED index_t silt::index_match<int>(const silt::tensor_t<int>& data, const int value);
template EXPORT_SHARED index_t silt::index_match<float>(const silt::tensor_t<float>& data, const float value);
template EXPORT_SHARED index_t silt::index_match<double>(const silt::tensor_t<double>& data, const double value);

} // end of namespace silt
