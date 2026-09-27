#include <iostream>
#include <silt/core/error.hpp>
#include <silt/core/operation.hpp>
#include <silt/op/common.hpp>

namespace silt {

//
// Unary Operations
//

// Set

template<typename T>
void set(tensor_t<T> lhs, const T rhs) {
  op::uniop_inplace(lhs, [rhs] GPU_ENABLE(const T a) {
    return rhs;
  });
}

template EXPORT_SHARED void silt::set<int>(silt::tensor_t<int> lhs, const int rhs);
template EXPORT_SHARED void silt::set<float>(silt::tensor_t<float> lhs, const float rhs);
template EXPORT_SHARED void silt::set<double>(silt::tensor_t<double> lhs, const double rhs);

template<typename T>
void set(view_t<T> lhs, const T rhs) {
  op::uniop_inplace(lhs, [rhs] GPU_ENABLE(const T a) {
    return rhs;
  });
}

template EXPORT_SHARED void silt::set<int>(silt::view_t<int> lhs, const int rhs);
template EXPORT_SHARED void silt::set<float>(silt::view_t<float> lhs, const float rhs);
template EXPORT_SHARED void silt::set<double>(silt::view_t<double> lhs, const double rhs);

// Add

template<typename T>
void add(tensor_t<T> lhs, const T rhs) {
  op::uniop_inplace(lhs, [rhs] GPU_ENABLE(const T a) {
    return a + rhs;
  });
}

template EXPORT_SHARED void silt::add<int>(silt::tensor_t<int> buffer, const int val);
template EXPORT_SHARED void silt::add<float>(silt::tensor_t<float> buffer, const float val);
template EXPORT_SHARED void silt::add<double>(silt::tensor_t<double> buffer, const double val);

template<typename T>
void add(view_t<T> lhs, const T rhs) {
  op::uniop_inplace(lhs, [rhs] GPU_ENABLE(const T a) {
    return a + rhs;
  });
}

template EXPORT_SHARED void silt::add<int>(silt::view_t<int> buffer, const int val);
template EXPORT_SHARED void silt::add<float>(silt::view_t<float> buffer, const float val);
template EXPORT_SHARED void silt::add<double>(silt::view_t<double> buffer, const double val);

// Multiply

template<typename T>
void multiply(tensor_t<T> lhs, const T rhs) {
  op::uniop_inplace(lhs, [rhs] GPU_ENABLE(const T a) {
    return a * rhs;
  });
}

template EXPORT_SHARED void silt::multiply<int>(silt::tensor_t<int> buffer, const int val);
template EXPORT_SHARED void silt::multiply<float>(silt::tensor_t<float> buffer, const float val);
template EXPORT_SHARED void silt::multiply<double>(silt::tensor_t<double> buffer, const double val);

template<typename T>
void multiply(view_t<T> lhs, const T rhs) {
  op::uniop_inplace(lhs, [rhs] GPU_ENABLE(const T a) {
    return a * rhs;
  });
}

template EXPORT_SHARED void silt::multiply<int>(silt::view_t<int> buffer, const int val);
template EXPORT_SHARED void silt::multiply<float>(silt::view_t<float> buffer, const float val);
template EXPORT_SHARED void silt::multiply<double>(silt::view_t<double> buffer, const double val);

// Divide

template<typename T>
void divide(tensor_t<T> lhs, const T rhs) {
  op::uniop_inplace(lhs, [rhs] GPU_ENABLE(const T a) {
    return a / rhs;
  });
}

template EXPORT_SHARED void silt::divide<int>(silt::tensor_t<int> buffer, const int val);
template EXPORT_SHARED void silt::divide<float>(silt::tensor_t<float> buffer, const float val);
template EXPORT_SHARED void silt::divide<double>(silt::tensor_t<double> buffer, const double val);

template<typename T>
void divide(view_t<T> lhs, const T rhs) {
  op::uniop_inplace(lhs, [rhs] GPU_ENABLE(const T a) {
    return a / rhs;
  });
}

template EXPORT_SHARED void silt::divide<int>(silt::view_t<int> buffer, const int val);
template EXPORT_SHARED void silt::divide<float>(silt::view_t<float> buffer, const float val);
template EXPORT_SHARED void silt::divide<double>(silt::view_t<double> buffer, const double val);

// Clamp

template<typename T>
void clamp(silt::tensor_t<T> lhs, const T min, const T max) {
  op::uniop_inplace(lhs, [min, max] GPU_ENABLE(const T a) {
    return glm::clamp(a, min, max);
  });
}

template EXPORT_SHARED void silt::clamp<int>(silt::tensor_t<int> buffer, const int min, const int max);
template EXPORT_SHARED void silt::clamp<float>(silt::tensor_t<float> buffer, const float min, const float max);
template EXPORT_SHARED void silt::clamp<double>(silt::tensor_t<double> buffer, const double min, const double max);

template<typename T>
void clamp(silt::view_t<T> lhs, const T min, const T max) {
  op::uniop_inplace(lhs, [min, max] GPU_ENABLE(const T a) {
    return glm::clamp(a, min, max);
  });
}

template EXPORT_SHARED void silt::clamp<int>(silt::view_t<int> buffer, const int min, const int max);
template EXPORT_SHARED void silt::clamp<float>(silt::view_t<float> buffer, const float min, const float max);
template EXPORT_SHARED void silt::clamp<double>(silt::view_t<double> buffer, const double min, const double max);

// Minimum

template<typename T>
void minimum(tensor_t<T> lhs, const T value) {
  op::uniop_inplace(lhs, [value] GPU_ENABLE(const T a) {
    return glm::min(a, value);
  });
}

template EXPORT_SHARED void silt::minimum<int>(silt::tensor_t<int> lhs, const int value);
template EXPORT_SHARED void silt::minimum<float>(silt::tensor_t<float> lhs, const float value);
template EXPORT_SHARED void silt::minimum<double>(silt::tensor_t<double> lhs, const double value);

template<typename T>
void minimum(view_t<T> lhs, const T value) {
  op::uniop_inplace(lhs, [value] GPU_ENABLE(const T a) {
    return glm::min(a, value);
  });
}

template EXPORT_SHARED void silt::minimum<int>(silt::view_t<int> lhs, const int value);
template EXPORT_SHARED void silt::minimum<float>(silt::view_t<float> lhs, const float value);
template EXPORT_SHARED void silt::minimum<double>(silt::view_t<double> lhs, const double value);

// Maximum

template<typename T>
void maximum(tensor_t<T> lhs, const T value) {
  op::uniop_inplace(lhs, [value] GPU_ENABLE(const T a) {
    return glm::max(a, value);
  });
}

template EXPORT_SHARED void silt::maximum<int>(silt::tensor_t<int> lhs, const int value);
template EXPORT_SHARED void silt::maximum<float>(silt::tensor_t<float> lhs, const float value);
template EXPORT_SHARED void silt::maximum<double>(silt::tensor_t<double> lhs, const double value);

template<typename T>
void maximum(view_t<T> lhs, const T value) {
  op::uniop_inplace(lhs, [value] GPU_ENABLE(const T a) {
    return glm::max(a, value);
  });
}

template EXPORT_SHARED void silt::maximum<int>(silt::view_t<int> lhs, const int value);
template EXPORT_SHARED void silt::maximum<float>(silt::view_t<float> lhs, const float value);
template EXPORT_SHARED void silt::maximum<double>(silt::view_t<double> lhs, const double value);

} // end of namespace silt
