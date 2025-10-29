#ifndef SILT_PYTHON_SHAPE
#define SILT_PYTHON_SHAPE

#include <nanobind/nanobind.h>
namespace nb = nanobind;

#include <silt/core/shape.hpp>
#include <format>

//! General Util Binding Function
void bind_shape(nb::module_& module) {

//
// Shape Type Binding
//

auto shape = nb::class_<silt::shape>(module, "shape");

shape.def(nb::init<>());
shape.def(nb::init<int>());
shape.def(nb::init<int, int>());
shape.def(nb::init<int, int, int>());
shape.def(nb::init<int, int, int, int>());

shape.def_ro("dim", &silt::shape::dim);
shape.def_ro("elem", &silt::shape::elem);

shape.def("__getitem__", &silt::shape::operator[]);

shape.def("__repr__", [](const silt::shape& shape){
  switch(shape.dim){
    case 1:
      return std::format("silt.shape({})", shape[0]).c_str(); 
    case 2:
      return std::format("silt.shape({}, {})", shape[0], shape[1]).c_str(); 
    case 3:
      return std::format("silt.shape({}, {}, {})", shape[0], shape[1], shape[2]).c_str(); 
    case 4:
      return std::format("silt.shape({}, {}, {}, {})", shape[0], shape[1], shape[2], shape[3]).c_str(); 
    default:
      return "silt.shape()";  
  }
});

}

#endif