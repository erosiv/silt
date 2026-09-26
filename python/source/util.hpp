#pragma once

#include <nanobind/nanobind.h>
#include <silt/core/error.hpp>
#include <silt/core/tensor.hpp>
namespace nb = nanobind;

namespace {

//
// Type Guard
//

//! Throw silt::error::mismatch_type unless `t` holds dtype `want`.
inline void require_type(const silt::tensor& t, silt::dtype want) {
  if (t.type() != want)
    throw silt::error::mismatch_type(want, t.type());
}

//
// Slice Unpacking
//

void __unpack_slice(
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

}
