#ifndef SILT_OP_COMMON_UNARY_CU
#define SILT_OP_COMMON_UNARY_CU
#define HAS_CUDA

#include <silt/op/common.hpp>
#include <silt/op/gather.hpp>
#include <silt/core/error.hpp>
#include <silt/core/operation.hpp>
#include <iostream>

namespace silt {

//
// Unary Operations
//

// Set

template<typename T>
void set(tensor_t<T> lhs, const T rhs) {
  op::uniop_inplace(lhs, [rhs] GPU_ENABLE (const T a){
    return rhs;
  });
}

template void silt::set<int>   (silt::tensor_t<int> lhs,     const int rhs);
template void silt::set<float> (silt::tensor_t<float> lhs,   const float rhs);
template void silt::set<double>(silt::tensor_t<double> lhs,  const double rhs);

template<typename T>
void set(view_t<T> lhs, const T rhs) {
  op::uniop_inplace(lhs, [rhs] GPU_ENABLE (const T a){
    return rhs;
  });
}

template void silt::set<int>   (silt::view_t<int> lhs,     const int rhs);
template void silt::set<float> (silt::view_t<float> lhs,   const float rhs);
template void silt::set<double>(silt::view_t<double> lhs,  const double rhs);

// Add

template<typename T>
void add(tensor_t<T> lhs, const T rhs) {
  op::uniop_inplace(lhs, [rhs] GPU_ENABLE (const T a){
    return a + rhs;
  });
}

template void silt::add<int>   (silt::tensor_t<int> buffer,    const int val);
template void silt::add<float> (silt::tensor_t<float> buffer,  const float val);
template void silt::add<double>(silt::tensor_t<double> buffer, const double val);

template<typename T>
void add(view_t<T> lhs, const T rhs) {
  op::uniop_inplace(lhs, [rhs] GPU_ENABLE (const T a){
    return a + rhs;
  });
}

template void silt::add<int>   (silt::view_t<int> buffer,    const int val);
template void silt::add<float> (silt::view_t<float> buffer,  const float val);
template void silt::add<double>(silt::view_t<double> buffer, const double val);

// Multiply

template<typename T>
void multiply(tensor_t<T> lhs, const T rhs) {
  op::uniop_inplace(lhs, [rhs] GPU_ENABLE (const T a){
    return a * rhs;
  });
}

template void silt::multiply<int>   (silt::tensor_t<int> buffer,    const int val);
template void silt::multiply<float> (silt::tensor_t<float> buffer,  const float val);
template void silt::multiply<double>(silt::tensor_t<double> buffer, const double val);

template<typename T>
void multiply(view_t<T> lhs, const T rhs) {
  op::uniop_inplace(lhs, [rhs] GPU_ENABLE (const T a){
    return a * rhs;
  });
}

template void silt::multiply<int>   (silt::view_t<int> buffer,    const int val);
template void silt::multiply<float> (silt::view_t<float> buffer,  const float val);
template void silt::multiply<double>(silt::view_t<double> buffer, const double val);

// Divide

template<typename T>
void divide(tensor_t<T> lhs, const T rhs) {
  op::uniop_inplace(lhs, [rhs] GPU_ENABLE (const T a){
    return a / rhs;
  });
}

template void silt::divide<int>   (silt::tensor_t<int> buffer,    const int val);
template void silt::divide<float> (silt::tensor_t<float> buffer,  const float val);
template void silt::divide<double>(silt::tensor_t<double> buffer, const double val);

template<typename T>
void divide(view_t<T> lhs, const T rhs) {
  op::uniop_inplace(lhs, [rhs] GPU_ENABLE (const T a){
    return a / rhs;
  });
}

template void silt::divide<int>   (silt::view_t<int> buffer,    const int val);
template void silt::divide<float> (silt::view_t<float> buffer,  const float val);
template void silt::divide<double>(silt::view_t<double> buffer, const double val);

// Clamp

template<typename T>
void clamp(silt::tensor_t<T> lhs, const T min, const T max) {
  op::uniop_inplace(lhs, [min, max] GPU_ENABLE (const T a){
    return glm::clamp(a, min, max);
  });
}

template void silt::clamp<int>   (silt::tensor_t<int> buffer,    const int min,     const int max);
template void silt::clamp<float> (silt::tensor_t<float> buffer,  const float min,   const float max);
template void silt::clamp<double>(silt::tensor_t<double> buffer, const double min,  const double max);

template<typename T>
void clamp(silt::view_t<T> lhs, const T min, const T max) {
  op::uniop_inplace(lhs, [min, max] GPU_ENABLE (const T a){
    return glm::clamp(a, min, max);
  });
}

template void silt::clamp<int>   (silt::view_t<int> buffer,    const int min,     const int max);
template void silt::clamp<float> (silt::view_t<float> buffer,  const float min,   const float max);
template void silt::clamp<double>(silt::view_t<double> buffer, const double min,  const double max);

// Clone

template<typename T>
tensor_t<T> clone(const tensor_t<T> rhs) {
  tensor_t<T> lhs(rhs.shape(), silt::GPU);
  op::binop_inplace(lhs, rhs, [] GPU_ENABLE (const T a, const T b){
    return b;
  });
  return lhs;
}

template tensor_t<int>    silt::clone<int>   (const silt::tensor_t<int> rhs);
template tensor_t<float>  silt::clone<float> (const silt::tensor_t<float> rhs);
template tensor_t<double> silt::clone<double>(const silt::tensor_t<double> rhs);

} // end of namespace silt

#endif