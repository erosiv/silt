#include <iostream>
#include <silt/core/error.hpp>
#include <silt/core/operation.hpp>
#include <silt/op/common.hpp>

namespace silt {

//
// Binary Operations
//

// Set

template<typename T>
void set(tensor_t<T> lhs, const tensor_t<T> rhs) {
  op::binop_inplace(lhs, rhs, [] GPU_ENABLE(const T a, const T b) {
    return b;
  });
}

template EXPORT_SHARED void silt::set<int>(silt::tensor_t<int> lhs, const silt::tensor_t<int> rhs);
template EXPORT_SHARED void silt::set<float>(silt::tensor_t<float> lhs, const silt::tensor_t<float> rhs);
template EXPORT_SHARED void silt::set<double>(silt::tensor_t<double> lhs, const silt::tensor_t<double> rhs);
// int64_t: needed by make_numpy/make_torch (interop.hpp) for index_t, not
// part of the primitive-gated elementwise ops above.
template EXPORT_SHARED void silt::set<int64_t>(silt::tensor_t<int64_t> lhs, const silt::tensor_t<int64_t> rhs);

// Add

template<typename T>
void add(tensor_t<T> lhs, const tensor_t<T> rhs) {
  op::binop_inplace(lhs, rhs, [] GPU_ENABLE(const T a, const T b) {
    return a + b;
  });
}

template EXPORT_SHARED void silt::add<int>(silt::tensor_t<int> lhs, const silt::tensor_t<int> rhs);
template EXPORT_SHARED void silt::add<float>(silt::tensor_t<float> lhs, const silt::tensor_t<float> rhs);
template EXPORT_SHARED void silt::add<double>(silt::tensor_t<double> lhs, const silt::tensor_t<double> rhs);

// Multiply

template<typename T>
void multiply(tensor_t<T> lhs, const tensor_t<T> rhs) {
  op::binop_inplace(lhs, rhs, [] GPU_ENABLE(const T a, const T b) {
    return a * b;
  });
}

template EXPORT_SHARED void silt::multiply<int>(silt::tensor_t<int> lhs, const silt::tensor_t<int> rhs);
template EXPORT_SHARED void silt::multiply<float>(silt::tensor_t<float> lhs, const silt::tensor_t<float> rhs);
template EXPORT_SHARED void silt::multiply<double>(silt::tensor_t<double> lhs, const silt::tensor_t<double> rhs);

// Divide

template<typename T>
void divide(tensor_t<T> lhs, const tensor_t<T> rhs) {
  op::binop_inplace(lhs, rhs, [] GPU_ENABLE(const T a, const T b) {
    return a / b;
  });
}

template EXPORT_SHARED void silt::divide<int>(silt::tensor_t<int> lhs, const silt::tensor_t<int> rhs);
template EXPORT_SHARED void silt::divide<float>(silt::tensor_t<float> lhs, const silt::tensor_t<float> rhs);
template EXPORT_SHARED void silt::divide<double>(silt::tensor_t<double> lhs, const silt::tensor_t<double> rhs);

// Mix

template<typename T>
void mix(tensor_t<T> lhs, const tensor_t<T> rhs, const float w) {
  op::binop_inplace(lhs, rhs, [w] GPU_ENABLE(const T a, const T b) {
    return (1.0f - w) * a + w * b;
  });
}

template EXPORT_SHARED void silt::mix<int>(silt::tensor_t<int> buffer, const silt::tensor_t<int> rhs, const float w);
template EXPORT_SHARED void silt::mix<float>(silt::tensor_t<float> buffer, const silt::tensor_t<float> rhs, const float w);
template EXPORT_SHARED void silt::mix<double>(silt::tensor_t<double> buffer, const silt::tensor_t<double> rhs, const float w);

// Minimum

template<typename T>
void minimum(tensor_t<T> lhs, const tensor_t<T> rhs) {
  op::binop_inplace(lhs, rhs, [] GPU_ENABLE(const T a, const T b) {
    return glm::min(a, b);
  });
}

template EXPORT_SHARED void silt::minimum<int>(silt::tensor_t<int> lhs, const silt::tensor_t<int> rhs);
template EXPORT_SHARED void silt::minimum<float>(silt::tensor_t<float> lhs, const silt::tensor_t<float> rhs);
template EXPORT_SHARED void silt::minimum<double>(silt::tensor_t<double> lhs, const silt::tensor_t<double> rhs);

// Maximum

template<typename T>
void maximum(tensor_t<T> lhs, const tensor_t<T> rhs) {
  op::binop_inplace(lhs, rhs, [] GPU_ENABLE(const T a, const T b) {
    return glm::max(a, b);
  });
}

template EXPORT_SHARED void silt::maximum<int>(silt::tensor_t<int> lhs, const silt::tensor_t<int> rhs);
template EXPORT_SHARED void silt::maximum<float>(silt::tensor_t<float> lhs, const silt::tensor_t<float> rhs);
template EXPORT_SHARED void silt::maximum<double>(silt::tensor_t<double> lhs, const silt::tensor_t<double> rhs);

} // end of namespace silt
