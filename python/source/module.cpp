#ifndef PYTHON_MODULE_ROOT
#define PYTHON_MODULE_ROOT

#include <nanobind/nanobind.h>
namespace nb = nanobind;

// Bind Function Declarations
void bind_component_A(nb::module_& module);

// Module Main Function
NB_MODULE(MODULE_NAME, module){

nb::set_leak_warnings(false);

module.doc() = "Python Module for my_project";

bind_component_A(module);

}

#endif