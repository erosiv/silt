#include <nanobind/nanobind.h>
namespace nb = nanobind;

#include <nanobind/make_iterator.h>
#include <nanobind/ndarray.h>

#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <silt/core/error.hpp>
#include <silt/core/types.hpp>

#include "glm.hpp"

//
//
//

//! General Util Binding Function
void bind_util(nb::module_& module) {

  //
  // Type Enumerator Binding
  //

  nb::enum_<silt::dtype>(module, "dtype")
      .value("int", silt::dtype::INT32)
      .value("float32", silt::dtype::FLOAT32)
      .value("float64", silt::dtype::FLOAT64)
      .value("rng", silt::dtype::RNG)
      .value("int64", silt::dtype::INT64)
      .export_values();

  //
  // Device Enumerator Binding
  //

  nb::enum_<silt::host_t>(module, "host")
      .value("cpu", silt::host_t::CPU)
      .value("gpu", silt::host_t::GPU)
      .export_values();
}
