#ifndef SILT_PYTHON_UTIL
#define SILT_PYTHON_UTIL

#include <nanobind/nanobind.h>
namespace nb = nanobind;

#include <nanobind/ndarray.h>
#include <nanobind/make_iterator.h>

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>

#include <silt/core/types.hpp>
#include <silt/core/error.hpp>

#include "glm.hpp"

//
//
//

//! General Util Binding Function
void bind_util(nb::module_& module){

//
// Type Enumerator Binding
//

nb::enum_<silt::dtype>(module, "dtype")
  .value("int", silt::dtype::INT)
  .value("float32", silt::dtype::FLOAT32)
  .value("float64", silt::dtype::FLOAT64)
  .export_values();

//
// Device Enumerator Binding
//

nb::enum_<silt::host_t>(module, "host")
  .value("cpu", silt::host_t::CPU)
  .value("gpu", silt::host_t::GPU)
  .export_values();

}

#endif