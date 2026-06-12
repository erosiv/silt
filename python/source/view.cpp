#ifndef SILT_PYTHON_VIEW
#define SILT_PYTHON_VIEW

#include <nanobind/nanobind.h>
namespace nb = nanobind;
using namespace nb::literals;

#include <silt/core/view.hpp>
#include <silt/core/error.hpp>
#include <format>

#include "glm.hpp"
#include "util.hpp"

//! General Util Binding Function
void bind_view(nb::module_& module) {

//
// View Type Binding:
//  Note that we deliberately do not expose the fine-detailed view manipulation functions,
//  so that views are primarily constructed directly from tensor types to avoid user error.
//

auto view = nb::class_<silt::view>(module, "view");

// Data Inspection

view.def_prop_ro("type", &silt::view::type);
view.def_prop_ro("elem", &silt::view::elem);
view.def_prop_ro("host", &silt::view::host);
view.def_prop_ro("slice", &silt::view::slice);

}

#endif