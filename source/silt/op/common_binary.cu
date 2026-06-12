#ifndef SILT_OP_COMMON_BINARY_CU
#define SILT_OP_COMMON_BINARY_CU
#define HAS_CUDA

#include <silt/op/common.hpp>
#include <silt/op/gather.hpp>
#include <silt/core/error.hpp>
#include <silt/core/operation.hpp>
#include <iostream>

namespace silt {

//
// Binary Operations
//

// Set

template<typename T>
void set(tensor_t<T> lhs, const tensor_t<T> rhs) {
  op::binop_inplace(lhs, rhs, [] GPU_ENABLE (const T a, const T b){
    return b;
  });
}

template void silt::set<int>   (silt::tensor_t<int> lhs,     const silt::tensor_t<int> rhs);
template void silt::set<float> (silt::tensor_t<float> lhs,   const silt::tensor_t<float> rhs);
template void silt::set<double>(silt::tensor_t<double> lhs,  const silt::tensor_t<double> rhs);

// Add

template<typename T>
void add(tensor_t<T> lhs, const tensor_t<T> rhs) {
  op::binop_inplace(lhs, rhs, [] GPU_ENABLE (const T a, const T b){
    return a + b;
  });
}

template void silt::add<int>   (silt::tensor_t<int> lhs,     const silt::tensor_t<int> rhs);
template void silt::add<float> (silt::tensor_t<float> lhs,   const silt::tensor_t<float> rhs);
template void silt::add<double>(silt::tensor_t<double> lhs,  const silt::tensor_t<double> rhs);

// Multiply

template<typename T>
void multiply(tensor_t<T> lhs, const tensor_t<T> rhs) {
  op::binop_inplace(lhs, rhs, [] GPU_ENABLE (const T a, const T b){
    return a * b;
  });
}

template void silt::multiply<int>   (silt::tensor_t<int> lhs,     const silt::tensor_t<int> rhs);
template void silt::multiply<float> (silt::tensor_t<float> lhs,   const silt::tensor_t<float> rhs);
template void silt::multiply<double>(silt::tensor_t<double> lhs,  const silt::tensor_t<double> rhs);

// Divide

template<typename T>
void divide(tensor_t<T> lhs, const tensor_t<T> rhs) {
  op::binop_inplace(lhs, rhs, [] GPU_ENABLE (const T a, const T b){
    return a / b;
  });
}

template void silt::divide<int>   (silt::tensor_t<int> lhs,     const silt::tensor_t<int> rhs);
template void silt::divide<float> (silt::tensor_t<float> lhs,   const silt::tensor_t<float> rhs);
template void silt::divide<double>(silt::tensor_t<double> lhs,  const silt::tensor_t<double> rhs);

// Mix

template<typename T>
void mix(tensor_t<T> lhs, const tensor_t<T> rhs, const float w) {
  op::binop_inplace(lhs, rhs, [w] GPU_ENABLE (const T a, const T b){
    return (1.0f - w) * a + w * b;
  });
}

template void silt::mix<int>   (silt::tensor_t<int> buffer,     const silt::tensor_t<int> rhs, const float w);
template void silt::mix<float> (silt::tensor_t<float> buffer,   const silt::tensor_t<float> rhs, const float w);
template void silt::mix<double>(silt::tensor_t<double> buffer,  const silt::tensor_t<double> rhs, const float w);

} // end of namespace silt

#endif