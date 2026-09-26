#include <nanobind/nanobind.h>
namespace nb = nanobind;
using namespace nb::literals;

#include <format>
#include <nanobind/stl/string.h>
#include <silt/core/error.hpp>
#include <silt/core/view.hpp>

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
  // Alias matching numpy/torch naming, alongside the existing .type.
  view.def_prop_ro("dtype", &silt::view::type);

  view.def("__repr__", [](const silt::view& view) -> std::string {
    const std::string slice_repr = nb::repr(nb::cast(view.slice())).c_str();
    return std::format(
        "silt.view(silt.{}, {}, silt.{})",
        silt::detail::dtype_name(view.type()),
        slice_repr,
        silt::detail::host_name(view.host())
    );
  });
}
