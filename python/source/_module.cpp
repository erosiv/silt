#ifndef SILT_PYTHON
#define SILT_PYTHON

// silt Python Bindings
// Nicholas McDonald 2025

#include <nanobind/nanobind.h>

// Bind Function Declarations

namespace nb = nanobind;
void bind_shape(nb::module_& module);
void bind_tensor(nb::module_& module);
void bind_op(nb::module_& module);
void bind_util(nb::module_& module);

// Module Main Function

NB_MODULE(MODULE_NAME, module){

nb::set_leak_warnings(false);

module.doc() = "silt python bindings";

bind_shape(module);
bind_tensor(module);
bind_op(module);
bind_util(module);

}

#endif