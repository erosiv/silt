#include <nanobind/nanobind.h>
namespace nb = nanobind;

#include <nanobind/stl/vector.h>

#include "glm.hpp"
#include <silt/core/view.hpp>
#include <silt/op/indexed.hpp>

namespace {

//! Type/host guard for an index-set argument. `L` is tensor or view.
template<typename L>
void require_index_set(const L& lhs, const silt::tensor& ind) {
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

  module.def("index_range", [](const silt::tensor& data, const nb::object lo, const nb::object hi) {
    return silt::select(data.type(), [&data, &lo, &hi]<silt::primitive S>() -> silt::tensor {
      return silt::tensor(silt::index_range<S>(data.as<S>(), nb::cast<S>(lo), nb::cast<S>(hi)));
    });
  });

  module.def("index_greater", [](const silt::tensor& data, const nb::object value) {
    return silt::select(data.type(), [&data, &value]<silt::primitive S>() -> silt::tensor {
      return silt::tensor(silt::index_greater<S>(data.as<S>(), nb::cast<S>(value)));
    });
  });

  module.def("index_lesser", [](const silt::tensor& data, const nb::object value) {
    return silt::select(data.type(), [&data, &value]<silt::primitive S>() -> silt::tensor {
      return silt::tensor(silt::index_lesser<S>(data.as<S>(), nb::cast<S>(value)));
    });
  });

  module.def("index_match", [](const silt::tensor& data, const nb::object value) {
    return silt::select(data.type(), [&data, &value]<silt::primitive S>() -> silt::tensor {
      return silt::tensor(silt::index_match<S>(data.as<S>(), nb::cast<S>(value)));
    });
  });

  module.def("index_slice", [](const silt::slice& s) {
    return silt::tensor(silt::index_slice(s));
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

  //
  // Gather / Scatter
  //  Index sets address the logical linear space of a tensor, or of a
  //  view's slice, so slicing a view picks the elements an index set
  //  refers to (e.g. t[:, :, c] for an index set over shape (x, y)).
  //

  module.def("gather", [](const silt::tensor& src, const silt::tensor& ind) {
    require_index_set(src, ind);
    return silt::select(src.type(), [&src, &ind]<silt::primitive S>() -> silt::tensor {
      return silt::tensor(silt::gather<S>(src.as<S>(), ind.as<int64_t>()));
    });
  });

  module.def("gather", [](const silt::view& src, const silt::tensor& ind) {
    require_index_set(src, ind);
    return silt::select(src.type(), [&src, &ind]<silt::primitive S>() -> silt::tensor {
      return silt::tensor(silt::gather<S>(src.as<S>(), ind.as<int64_t>()));
    });
  });

  module.def("scatter_", [](silt::tensor& dst, const silt::tensor& src, const silt::tensor& ind) {
    require_index_set(dst, ind);
    if (dst.type() != src.type())
      throw silt::error::mismatch_type(dst.type(), src.type());
    silt::select(dst.type(), [&dst, &src, &ind]<silt::primitive S>() {
      silt::scatter<S>(dst.as<S>(), src.as<S>(), ind.as<int64_t>());
    });
  });

  module.def("scatter_", [](silt::view& dst, const silt::tensor& src, const silt::tensor& ind) {
    require_index_set(dst, ind);
    if (dst.type() != src.type())
      throw silt::error::mismatch_type(dst.type(), src.type());
    silt::select(dst.type(), [&dst, &src, &ind]<silt::primitive S>() {
      silt::scatter<S>(dst.as<S>(), src.as<S>(), ind.as<int64_t>());
    });
  });

  //
  // Indexed Operations (View)
  //  The index set addresses the slice-space of the view.
  //

  module.def("indexed_set", [](silt::view& lhs, const nb::object value, const silt::tensor& ind) {
    require_index_set(lhs, ind);
    silt::select(lhs.type(), [&lhs, &value, &ind]<silt::primitive S>() {
      silt::indexed_set<S>(lhs.as<S>(), nb::cast<S>(value), ind.as<int64_t>());
    });
  });

  module.def("indexed_add_", [](silt::view& lhs, const silt::view& rhs, const silt::tensor& ind) {
    require_index_set(lhs, ind);
    if (lhs.type() != rhs.type())
      throw silt::error::mismatch_type(lhs.type(), rhs.type());
    if (lhs.elem() != rhs.elem())
      throw silt::error::mismatch_size(lhs.elem(), rhs.elem());
    silt::select(lhs.type(), [&lhs, &rhs, &ind]<silt::primitive S>() {
      silt::indexed_add<S>(lhs.as<S>(), rhs.as<S>(), ind.as<int64_t>());
    });
  });

  module.def("indexed_add_", [](silt::view& lhs, const nb::object rhs, const silt::tensor& ind) {
    require_index_set(lhs, ind);
    silt::select(lhs.type(), [&lhs, &rhs, &ind]<silt::primitive S>() {
      silt::indexed_add<S>(lhs.as<S>(), nb::cast<S>(rhs), ind.as<int64_t>());
    });
  });

  module.def("indexed_multiply_", [](silt::view& lhs, const silt::view& rhs, const silt::tensor& ind) {
    require_index_set(lhs, ind);
    if (lhs.type() != rhs.type())
      throw silt::error::mismatch_type(lhs.type(), rhs.type());
    if (lhs.elem() != rhs.elem())
      throw silt::error::mismatch_size(lhs.elem(), rhs.elem());
    silt::select(lhs.type(), [&lhs, &rhs, &ind]<silt::primitive S>() {
      silt::indexed_multiply<S>(lhs.as<S>(), rhs.as<S>(), ind.as<int64_t>());
    });
  });

  module.def("indexed_multiply_", [](silt::view& lhs, const nb::object rhs, const silt::tensor& ind) {
    require_index_set(lhs, ind);
    silt::select(lhs.type(), [&lhs, &rhs, &ind]<silt::primitive S>() {
      silt::indexed_multiply<S>(lhs.as<S>(), nb::cast<S>(rhs), ind.as<int64_t>());
    });
  });

  module.def("indexed_divide_", [](silt::view& lhs, const silt::view& rhs, const silt::tensor& ind) {
    require_index_set(lhs, ind);
    if (lhs.type() != rhs.type())
      throw silt::error::mismatch_type(lhs.type(), rhs.type());
    if (lhs.elem() != rhs.elem())
      throw silt::error::mismatch_size(lhs.elem(), rhs.elem());
    silt::select(lhs.type(), [&lhs, &rhs, &ind]<silt::primitive S>() {
      silt::indexed_divide<S>(lhs.as<S>(), rhs.as<S>(), ind.as<int64_t>());
    });
  });

  module.def("indexed_divide_", [](silt::view& lhs, const nb::object rhs, const silt::tensor& ind) {
    require_index_set(lhs, ind);
    silt::select(lhs.type(), [&lhs, &rhs, &ind]<silt::primitive S>() {
      silt::indexed_divide<S>(lhs.as<S>(), nb::cast<S>(rhs), ind.as<int64_t>());
    });
  });

  module.def("indexed_mix_", [](silt::view& lhs, const silt::view& rhs, const silt::tensor& ind, const float w) {
    require_index_set(lhs, ind);
    if (lhs.type() != rhs.type())
      throw silt::error::mismatch_type(lhs.type(), rhs.type());
    if (lhs.elem() != rhs.elem())
      throw silt::error::mismatch_size(lhs.elem(), rhs.elem());
    silt::select(lhs.type(), [&lhs, &rhs, &ind, w]<silt::primitive S>() {
      silt::indexed_mix<S>(lhs.as<S>(), rhs.as<S>(), ind.as<int64_t>(), w);
    });
  });

  //
  // Indexed Reductions
  //

  module.def("indexed_sum", [](const silt::tensor& lhs, const silt::tensor& ind) {
    require_index_set(lhs, ind);
    return silt::select(lhs.type(), [&lhs, &ind]<silt::primitive S>() -> nb::object {
      return nb::cast(silt::indexed_sum<S>(lhs.as<S>(), ind.as<int64_t>()));
    });
  });

  module.def("indexed_mean", [](const silt::tensor& lhs, const silt::tensor& ind) {
    require_index_set(lhs, ind);
    return silt::select(lhs.type(), [&lhs, &ind]<silt::primitive S>() -> nb::object {
      return nb::cast(silt::indexed_mean<S>(lhs.as<S>(), ind.as<int64_t>()));
    });
  });

  module.def("indexed_min", [](const silt::tensor& lhs, const silt::tensor& ind) {
    require_index_set(lhs, ind);
    return silt::select(lhs.type(), [&lhs, &ind]<silt::primitive S>() -> nb::object {
      return nb::cast(silt::indexed_min<S>(lhs.as<S>(), ind.as<int64_t>()));
    });
  });

  module.def("indexed_max", [](const silt::tensor& lhs, const silt::tensor& ind) {
    require_index_set(lhs, ind);
    return silt::select(lhs.type(), [&lhs, &ind]<silt::primitive S>() -> nb::object {
      return nb::cast(silt::indexed_max<S>(lhs.as<S>(), ind.as<int64_t>()));
    });
  });

  module.def("indexed_argmin", [](const silt::tensor& lhs, const silt::tensor& ind) {
    require_index_set(lhs, ind);
    return silt::select(lhs.type(), [&lhs, &ind]<silt::primitive S>() -> nb::object {
      return nb::cast(silt::indexed_argmin<S>(lhs.as<S>(), ind.as<int64_t>()));
    });
  });

  module.def("indexed_argmax", [](const silt::tensor& lhs, const silt::tensor& ind) {
    require_index_set(lhs, ind);
    return silt::select(lhs.type(), [&lhs, &ind]<silt::primitive S>() -> nb::object {
      return nb::cast(silt::indexed_argmax<S>(lhs.as<S>(), ind.as<int64_t>()));
    });
  });

  //
  // Indexed Reductions (View)
  //

  module.def("indexed_sum", [](const silt::view& lhs, const silt::tensor& ind) {
    require_index_set(lhs, ind);
    return silt::select(lhs.type(), [&lhs, &ind]<silt::primitive S>() -> nb::object {
      return nb::cast(silt::indexed_sum<S>(lhs.as<S>(), ind.as<int64_t>()));
    });
  });

  module.def("indexed_mean", [](const silt::view& lhs, const silt::tensor& ind) {
    require_index_set(lhs, ind);
    return silt::select(lhs.type(), [&lhs, &ind]<silt::primitive S>() -> nb::object {
      return nb::cast(silt::indexed_mean<S>(lhs.as<S>(), ind.as<int64_t>()));
    });
  });

  module.def("indexed_min", [](const silt::view& lhs, const silt::tensor& ind) {
    require_index_set(lhs, ind);
    return silt::select(lhs.type(), [&lhs, &ind]<silt::primitive S>() -> nb::object {
      return nb::cast(silt::indexed_min<S>(lhs.as<S>(), ind.as<int64_t>()));
    });
  });

  module.def("indexed_max", [](const silt::view& lhs, const silt::tensor& ind) {
    require_index_set(lhs, ind);
    return silt::select(lhs.type(), [&lhs, &ind]<silt::primitive S>() -> nb::object {
      return nb::cast(silt::indexed_max<S>(lhs.as<S>(), ind.as<int64_t>()));
    });
  });

  module.def("indexed_argmin", [](const silt::view& lhs, const silt::tensor& ind) {
    require_index_set(lhs, ind);
    return silt::select(lhs.type(), [&lhs, &ind]<silt::primitive S>() -> nb::object {
      return nb::cast(silt::indexed_argmin<S>(lhs.as<S>(), ind.as<int64_t>()));
    });
  });

  module.def("indexed_argmax", [](const silt::view& lhs, const silt::tensor& ind) {
    require_index_set(lhs, ind);
    return silt::select(lhs.type(), [&lhs, &ind]<silt::primitive S>() -> nb::object {
      return nb::cast(silt::indexed_argmax<S>(lhs.as<S>(), ind.as<int64_t>()));
    });
  });
}
