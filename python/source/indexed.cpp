#include <nanobind/nanobind.h>
namespace nb = nanobind;

#include <nanobind/stl/vector.h>

#include "glm.hpp"
#include <silt/op/indexed.hpp>

namespace {

//! Type/host guard for an index-set argument.
void require_index_set(const silt::tensor& lhs, const silt::tensor& ind) {
  if (ind.type() != silt::dtype::INT64)
    throw silt::error::mismatch_type(silt::dtype::INT64, ind.type());
  if (lhs.host() != ind.host())
    throw silt::error::mismatch_host(lhs.host(), ind.host());
}

} // namespace

void bind_indexed(nb::module_& module) {

  //
  // Index-Set Construction
  //

  module.def("index_radius", [](const silt::shape& shape, const silt::vec2 center, const float radius) {
    return silt::tensor(silt::index_radius(shape, center, radius));
  });

  module.def("index_box", [](const silt::shape& shape, const silt::vec2 lo, const silt::vec2 hi) {
    return silt::tensor(silt::index_box(shape, lo, hi));
  });

  module.def("index_polygon", [](const silt::shape& shape, const std::vector<silt::vec2>& vertices) {
    silt::tensor_t<silt::vec2> verts(silt::shape((int)vertices.size()), silt::host_t::CPU);
    for (size_t i = 0; i < vertices.size(); ++i)
      verts[i] = vertices[i];
    verts.to_gpu();
    return silt::tensor(silt::index_polygon(shape, verts));
  });

  //
  // Indexed Operations
  //

  module.def("indexed_set", [](silt::tensor& lhs, const nb::object value, const silt::tensor& ind) {
    require_index_set(lhs, ind);
    silt::select(lhs.type(), [&lhs, &value, &ind]<silt::primitive S>() {
      silt::indexed_set<S>(lhs.as<S>(), nb::cast<S>(value), ind.as<int64_t>());
    });
  });

  module.def("indexed_add_", [](silt::tensor& lhs, const silt::tensor& rhs, const silt::tensor& ind) {
    require_index_set(lhs, ind);
    if (lhs.type() != rhs.type())
      throw silt::error::mismatch_type(lhs.type(), rhs.type());
    if (lhs.elem() != rhs.elem())
      throw silt::error::mismatch_size(lhs.elem(), rhs.elem());
    silt::select(lhs.type(), [&lhs, &rhs, &ind]<silt::primitive S>() {
      silt::indexed_add<S>(lhs.as<S>(), rhs.as<S>(), ind.as<int64_t>());
    });
  });

  module.def("indexed_add_", [](silt::tensor& lhs, const nb::object rhs, const silt::tensor& ind) {
    require_index_set(lhs, ind);
    silt::select(lhs.type(), [&lhs, &rhs, &ind]<silt::primitive S>() {
      silt::indexed_add<S>(lhs.as<S>(), nb::cast<S>(rhs), ind.as<int64_t>());
    });
  });

  module.def("indexed_multiply_", [](silt::tensor& lhs, const silt::tensor& rhs, const silt::tensor& ind) {
    require_index_set(lhs, ind);
    if (lhs.type() != rhs.type())
      throw silt::error::mismatch_type(lhs.type(), rhs.type());
    if (lhs.elem() != rhs.elem())
      throw silt::error::mismatch_size(lhs.elem(), rhs.elem());
    silt::select(lhs.type(), [&lhs, &rhs, &ind]<silt::primitive S>() {
      silt::indexed_multiply<S>(lhs.as<S>(), rhs.as<S>(), ind.as<int64_t>());
    });
  });

  module.def("indexed_multiply_", [](silt::tensor& lhs, const nb::object rhs, const silt::tensor& ind) {
    require_index_set(lhs, ind);
    silt::select(lhs.type(), [&lhs, &rhs, &ind]<silt::primitive S>() {
      silt::indexed_multiply<S>(lhs.as<S>(), nb::cast<S>(rhs), ind.as<int64_t>());
    });
  });

  module.def("indexed_divide_", [](silt::tensor& lhs, const silt::tensor& rhs, const silt::tensor& ind) {
    require_index_set(lhs, ind);
    if (lhs.type() != rhs.type())
      throw silt::error::mismatch_type(lhs.type(), rhs.type());
    if (lhs.elem() != rhs.elem())
      throw silt::error::mismatch_size(lhs.elem(), rhs.elem());
    silt::select(lhs.type(), [&lhs, &rhs, &ind]<silt::primitive S>() {
      silt::indexed_divide<S>(lhs.as<S>(), rhs.as<S>(), ind.as<int64_t>());
    });
  });

  module.def("indexed_divide_", [](silt::tensor& lhs, const nb::object rhs, const silt::tensor& ind) {
    require_index_set(lhs, ind);
    silt::select(lhs.type(), [&lhs, &rhs, &ind]<silt::primitive S>() {
      silt::indexed_divide<S>(lhs.as<S>(), nb::cast<S>(rhs), ind.as<int64_t>());
    });
  });

  module.def("indexed_mix_", [](silt::tensor& lhs, const silt::tensor& rhs, const silt::tensor& ind, const float w) {
    require_index_set(lhs, ind);
    if (lhs.type() != rhs.type())
      throw silt::error::mismatch_type(lhs.type(), rhs.type());
    if (lhs.elem() != rhs.elem())
      throw silt::error::mismatch_size(lhs.elem(), rhs.elem());
    silt::select(lhs.type(), [&lhs, &rhs, &ind, w]<silt::primitive S>() {
      silt::indexed_mix<S>(lhs.as<S>(), rhs.as<S>(), ind.as<int64_t>(), w);
    });
  });
}
