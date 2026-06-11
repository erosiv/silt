#ifndef SILT_PYTHON_VIEW
#define SILT_PYTHON_VIEW

#include <nanobind/nanobind.h>
namespace nb = nanobind;
using namespace nb::literals;

#include <silt/core/view.hpp>
#include <silt/core/error.hpp>
#include <format>

#include "glm.hpp"

//! General Util Binding Function
void bind_view(nb::module_& module) {

//
// View Type Binding
//

auto view = nb::class_<silt::view>(module, "view");

// Data Inspection

view.def_prop_ro("type", &silt::view::type);
view.def_prop_ro("elem", &silt::view::elem);
view.def_prop_ro("host", &silt::view::host);
view.def_prop_ro("slice", &silt::view::slice);

view.def("reshape", [](silt::view& view, int d0, int d1, int d2, int d3) {
  view.reshape(d0, d1, d2, d3);
  return view;
}, "d0"_a = 1, "d1"_a = 1, "d2"_a = 1, "d3"_a = 1);

view.def("index", [](silt::view& view, int dim, int offset, int stride, int extent) {
  view.index(dim, offset, stride, extent);
  return view;
});

view.def("reset", [](silt::view& view){
  view.reset();
  return view;
});

//
// Slicing Logic
//

view.def("__getitem__", [](silt::view& view, nb::tuple tuple) -> silt::view {

  // Validate Number of Dimensions
  const size_t size = tuple.size();
  const auto shape = view.slice().shape();
  if(size != view.slice().dim()) {
    throw silt::error::mismatch_size(shape.dim(), size);
  }

  // Iterate over Subscript Tuple
  for(size_t d = 0; d < size; ++d) {

    nb::handle handle = tuple[d];
    Py_ssize_t offset, stride, extent;

    if (PySlice_Check(handle.ptr())) {

      if (PySlice_Unpack(handle.ptr(), &offset, &extent, &stride) < 0) {
        throw nb::python_error();
      }

      if(offset >= shape.ext()[d]) {
        throw silt::error::out_of_bounds(offset, shape.ext()[d]);
      }

      extent = std::min((shape.ext()[d] - offset) / stride, extent);

    } else {

      offset = nb::cast<Py_ssize_t>(handle);
      if(offset >= shape.ext()[d]) {
        throw silt::error::out_of_bounds(offset, shape.ext()[d]);
      }

      stride = 1;
      extent = 1;

    }

    view.index(d, offset, stride, extent);

  }
  
  return view;

});

}

#endif