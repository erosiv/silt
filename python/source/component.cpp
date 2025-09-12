#ifndef PYTHON_MODULE_COMPONENT_A
#define PYTHON_MODULE_COMPONENT_A

#include <nanobind/nanobind.h>
namespace nb = nanobind;

//
// Specify Component Bindings
//

#include <my_cuda_lib/my_header.hpp>

void bind_component_A(nb::module_& module){

module.def("test", [](){
  my_project::my_func();
});

}

#endif