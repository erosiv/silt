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

} // end of namespace silt
