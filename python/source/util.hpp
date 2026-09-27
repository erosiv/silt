#pragma once

#include <nanobind/nanobind.h>
#include <silt/core/error.hpp>
#include <silt/core/tensor.hpp>
namespace nb = nanobind;

namespace silt {
namespace detail {

//
// Type Guard
//

//! Throw silt::error::mismatch_type unless `t` holds dtype `want`.
inline void require_type(const silt::tensor& t, silt::dtype want) {
  if (t.type() != want)
    throw silt::error::mismatch_type(want, t.type());
}

//
// Names for __repr__
//

//! The Python-facing name for a dtype, i.e. the name it is bound under
//! in util.cpp's `nb::enum_<dtype>` (not `typedesc<T>::name`, which uses
//! different strings for the C++-side error messages).
inline const char* dtype_name(const silt::dtype type) {
  switch (type) {
  case silt::INT32:
    return "int";
  case silt::FLOAT32:
    return "float32";
  case silt::FLOAT64:
    return "float64";
  case silt::RNG:
    return "rng";
  case silt::INT64:
    return "int64";
  default:
    return "unknown";
  }
}

//! The Python-facing name for a host, matching util.cpp's `nb::enum_<host_t>`.
inline const char* host_name(const silt::host_t host) {
  return host == silt::CPU ? "cpu" : "gpu";
}

//
// Slice Unpacking
//

// A double-underscore name is reserved to the implementation at any
// scope, namespace or not, so this stays plain and lives in `detail`
// instead. inline because util.hpp is included into three separate
// TUs (op.cpp, tensor.cpp, view.cpp): unlike the previous anonymous
// namespace, a named namespace does not give this internal linkage on
// its own.
inline void unpack_slice(
    nb::handle& handle,
    Py_ssize_t& offset,
    Py_ssize_t& stride,
    Py_ssize_t& extent
) {

  if (PySlice_Check(handle.ptr())) {
    if (PySlice_Unpack(handle.ptr(), &offset, &extent, &stride) < 0) {
      throw nb::python_error();
    }
  }

  else {
    offset = nb::cast<Py_ssize_t>(handle);
    stride = 1;
    extent = 1;
  }
}

} // namespace detail
} // namespace silt
