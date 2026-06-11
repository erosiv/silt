#ifndef SILT_PYTHON_SLICE
#define SILT_PYTHON_SLICE

#include <nanobind/nanobind.h>
namespace nb = nanobind;

#include <silt/core/slice.hpp>
#include <silt/core/error.hpp>
#include <format>

#include "glm.hpp"

//! General Util Binding Function
void bind_slice(nb::module_& module) {

//
// Shape Type Binding
//

auto slice = nb::class_<silt::slice>(module, "slice");

slice.def(nb::init<>());
slice.def(nb::init<int>());
slice.def(nb::init<int, int>());
slice.def(nb::init<int, int, int>());
slice.def(nb::init<int, int, int, int>());

slice.def_prop_ro("dim", &silt::slice::dim);
slice.def_prop_ro("shape", &silt::slice::shape);
slice.def_prop_ro("maxelem", &silt::slice::maxelem);
slice.def_prop_ro("elem", &silt::slice::elem);
slice.def_prop_ro("offset", &silt::slice::offset);
slice.def_prop_ro("stride", &silt::slice::stride);
slice.def_prop_ro("extent", &silt::slice::extent);

slice.def("reset", &silt::slice::reset);
slice.def("transform", &silt::slice::transform);

// slice.def("__repr__", [](const silt::slice& slice){
//   switch(slice.dim()){
//     case 1:
//       return std::format("silt.slice({})", slice[0]).c_str(); 
//     case 2:
//       return std::format("silt.slice({}, {})", slice[0], slice[1]).c_str(); 
//     case 3:
//       return std::format("silt.slice({}, {}, {})", slice[0], slice[1], slice[2]).c_str(); 
//     case 4:
//       return std::format("silt.slice({}, {}, {}, {})", slice[0], slice[1], slice[2], slice[3]).c_str(); 
//     default:
//       return "silt.slice()";  
//   }
// });

//
// Slicing Logic
//

/*
shape.def("__getitem__", [](const silt::shape& shape, nb::tuple tuple) -> silt::shape {

  // Validate Number of Dimensions
  const size_t size = tuple.size();
  if(size != shape.dim) {
    throw silt::error::mismatch_size(shape.dim, size);
  }

  // Copy Original Shape
  silt::shape out = shape;

  // Iterate over Subscript Tuple
  for(size_t d = 0; d < size; ++d) {

    nb::handle handle = tuple[d];
    Py_ssize_t offset, stride, extlim;

    if (PySlice_Check(handle.ptr())) {

      if (PySlice_Unpack(handle.ptr(), &offset, &extlim, &stride) < 0) {
        throw nb::python_error();
      }

      if(offset >= shape.ext[d]) {
        throw silt::error::out_of_bounds(offset, shape.ext[d]);
      }

      extlim = std::min(shape.ext[d] / stride, extlim);

    } else {

      offset = nb::cast<Py_ssize_t>(handle);
      if(offset >= shape.ext[d]) {
        throw silt::error::out_of_bounds(offset, shape.ext[d]);
      }

      stride = shape.ext[d];
      extlim = 1;

    }

//    out.offset[d] = offset;
//    out.stride[d] = stride;
//    out.extlim[d] = extlim;

  }
  
  return out;

});
*/

}

#endif