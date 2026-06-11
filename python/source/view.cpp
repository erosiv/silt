#ifndef SILT_PYTHON_VIEW
#define SILT_PYTHON_VIEW

#include <nanobind/nanobind.h>
namespace nb = nanobind;

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

//shape.def_ro("ext", &silt::shape::ext);
//shape.def_ro("offset", &silt::shape::offset);
//shape.def_ro("stride", &silt::shape::stride);
//shape.def_ro("extlim", &silt::shape::extlim);

// shape.def("__getitem__", &silt::shape::operator[]);
// 
// shape.def("index", &silt::shape::index);
// 
// shape.def("__repr__", [](const silt::shape& shape){
//   switch(shape.dim){
//     case 1:
//       return std::format("silt.shape({})", shape[0]).c_str(); 
//     case 2:
//       return std::format("silt.shape({}, {})", shape[0], shape[1]).c_str(); 
//     case 3:
//       return std::format("silt.shape({}, {}, {})", shape[0], shape[1], shape[2]).c_str(); 
//     case 4:
//       return std::format("silt.shape({}, {}, {}, {})", shape[0], shape[1], shape[2], shape[3]).c_str(); 
//     default:
//       return "silt.shape()";  
//   }
// });

//
// Slicing Logic
//

// shape.def("__getitem__", [](const silt::shape& shape, nb::tuple tuple) -> silt::shape {
// 
//   // Validate Number of Dimensions
//   const size_t size = tuple.size();
//   if(size != shape.dim) {
//     throw silt::error::mismatch_size(shape.dim, size);
//   }
// 
//   // Copy Original Shape
//   silt::shape out = shape;
// 
//   // Iterate over Subscript Tuple
//   for(size_t d = 0; d < size; ++d) {
// 
//     nb::handle handle = tuple[d];
//     Py_ssize_t offset, stride, extlim;
// 
//     if (PySlice_Check(handle.ptr())) {
// 
//       if (PySlice_Unpack(handle.ptr(), &offset, &extlim, &stride) < 0) {
//         throw nb::python_error();
//       }
// 
//       if(offset >= shape.ext[d]) {
//         throw silt::error::out_of_bounds(offset, shape.ext[d]);
//       }
// 
//       extlim = std::min(shape.ext[d] / stride, extlim);
// 
//     } else {
// 
//       offset = nb::cast<Py_ssize_t>(handle);
//       if(offset >= shape.ext[d]) {
//         throw silt::error::out_of_bounds(offset, shape.ext[d]);
//       }
// 
//       stride = shape.ext[d];
//       extlim = 1;
// 
//     }
// 
//     out.offset[d] = offset;
//     out.stride[d] = stride;
//     out.extlim[d] = extlim;
// 
//   }
//   
//   return out;
// 
// });

}

#endif