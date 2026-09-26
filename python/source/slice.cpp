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

}
