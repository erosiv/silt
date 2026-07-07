#ifndef SILT_OP_INDEXED_CU
#define SILT_OP_INDEXED_CU
#define HAS_CUDA

#include <silt/op/indexed.hpp>
#include <silt/core/operation.hpp>

namespace silt {

//
// Indexed Assignment Function
//

template<typename T>
void indexed_set(tensor_t<T> lhs, const T rhs, const tensor_t<int> ind) {
  op::uniop_inplace_indexed(lhs, ind, [rhs] GPU_ENABLE (const T a){
    return rhs;
  });
}

template EXPORT_SHARED void silt::indexed_set<int>   (silt::tensor_t<int> lhs,     const int rhs,     const tensor_t<int> ind);
template EXPORT_SHARED void silt::indexed_set<float> (silt::tensor_t<float> lhs,   const float rhs,   const tensor_t<int> ind);
template EXPORT_SHARED void silt::indexed_set<double>(silt::tensor_t<double> lhs,  const double rhs,  const tensor_t<int> ind);

//
// Index Generation Functions
//

__global__ void __index_radius(silt::tensor_t<int> index, const silt::shape shape, const silt::vec2 center, const float rad) {\
  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n < shape.elem()) {
    const silt::vec2 pos = shape.unflatten(n);
    if(glm::length(pos - center) < rad) {
      index[n] = n;
    } else {
      index[n] = shape.elem();
    }
  }
}

tensor_t<int> index_radius(const silt::shape shape, const silt::vec2 center, const float rad) {
  silt::tensor_t<int> index(shape, silt::host_t::GPU);
  __index_radius<<<block(shape.elem(), 512), 512>>>(index, shape, center, rad);
  return index;
}

} // end of namespace soil

#endif