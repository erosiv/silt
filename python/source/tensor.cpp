#include <nanobind/nanobind.h>
namespace nb = nanobind;
using namespace nb::literals;

#include "interop.hpp"
#include "util.hpp"
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <silt/core/tensor.hpp>
#include <silt/core/view.hpp>
#include <silt/op/common.hpp>

namespace silt {
namespace detail {

// Behind tensor.cpp's __getitem__/slice bindings; not part of the public
// API surface, hence `detail` and no leading double-underscore.
silt::view getitem_impl(silt::tensor& tensor, nb::tuple tuple) {
  return silt::select(tensor.type(), [&tensor, tuple]<typename T>() -> silt::view {
    // Construct a View from the Tensor:
    auto tensor_t = tensor.as<T>();
    auto view_t = tensor_t.template view<T>();
    auto shape = tensor_t.shape();
    // by default, the view adopts the tensor's shape
    view_t.reshape(shape[0], shape[1], shape[2], shape[3]);

    // Validate Number of Dimensions
    const size_t size = tuple.size();
    if (size != shape.dim()) {
      throw silt::error::mismatch_size(shape.dim(), size);
    }

    // Slice the View: unpack raw offset/stride/extent and forward to
    // view_t::index, which forwards to slice::index -- the single place
    // slicing is clamped and validated (see slice.hpp).
    for (size_t d = 0; d < size; ++d) {

      nb::handle handle = tuple[d];
      Py_ssize_t offset, stride, extent;
      unpack_slice(handle, shape[d], offset, stride, extent);
      view_t.index(d, offset, stride, extent);
    }

    // Keep the source tensor alive for as long as this view is (see the
    // view_t lifetime note in view.hpp): otherwise `del t` right after
    // `v = t[...]` leaves v pointing at freed memory.
    // silt::tensor's copy constructor refcount-shares the underlying
    // tensor_t<T> (see tensor.hpp), so this is a cheap owning handle,
    // not a deep copy.
    return silt::view(view_t, tensor);
  });
}

// Shared by the "numpy" and "__array__" bindings below.
nb::object tensor_to_numpy(const silt::tensor& tensor) {
  if (tensor.host() != silt::host_t::CPU)
    throw silt::error::unsupported_host(silt::host_t::CPU, tensor.host());
  return silt::select(tensor.type(), [&tensor]<typename T>() -> nb::object {
    if constexpr (nb::detail::is_ndarray_scalar_v<T>) {
      return silt::detail::make_numpy(tensor.as<T>());
    } else {
      throw std::invalid_argument("tensor type cannot be converted");
    }
  });
}

} // namespace detail
} // namespace silt

//! General Util Binding Function
void bind_tensor(nb::module_& module) {

  //
  // Tensor Type Binding
  //

  auto tensor = nb::class_<silt::tensor>(module, "tensor");
  tensor.def(nb::init<>());
  tensor.def(nb::init<const silt::dtype, const silt::shape>());
  tensor.def(nb::init<const silt::dtype, const silt::shape, const silt::host_t>());

  // Data Inspection

  tensor.def_prop_ro("type", &silt::tensor::type);
  tensor.def_prop_ro("elem", &silt::tensor::elem);
  tensor.def_prop_ro("size", &silt::tensor::size);
  // .size means bytes here (not element count, unlike numpy's .size) --
  // .elem is the descriptive name for element count. .nbytes is an alias
  // for people coming from numpy/torch, where .size means something else.
  tensor.def_prop_ro("nbytes", &silt::tensor::size);
  tensor.def_prop_ro("host", &silt::tensor::host);
  tensor.def_prop_ro("shape", &silt::tensor::shape);
  // Aliases matching numpy/torch naming, alongside the existing .type/.shape.
  tensor.def_prop_ro("dtype", &silt::tensor::type);
  tensor.def_prop_ro("ndim", [](const silt::tensor& tensor) { return tensor.shape().dim(); });

  tensor.def("__repr__", [](const silt::tensor& tensor) -> std::string {
    const std::string shape_repr = nb::repr(nb::cast(tensor.shape())).c_str();
    return std::format(
        "silt.tensor(silt.{}, {}, silt.{}, refs={})",
        silt::detail::dtype_name(tensor.type()),
        shape_repr,
        silt::detail::host_name(tensor.host()),
        tensor.refs()
    );
  });

  tensor.def("__len__", [](const silt::tensor& tensor) -> size_t {
    return (size_t)tensor.shape()[0];
  });

  // Iterates over the first dimension, same as a numpy array. Eagerly
  // builds each sub-view up front rather than lazily -- simpler than a
  // custom C++ iterator type, and tensors are not expected to be huge
  // along their leading dimension.
  tensor.def("__iter__", [](silt::tensor& tensor) {
    nb::list items;
    const size_t n = (size_t)tensor.shape()[0];
    for (size_t i = 0; i < n; ++i) {
      items.append(silt::detail::getitem_impl(tensor, nb::make_tuple((Py_ssize_t)i)));
    }
    return nb::iter(items);
  });

  tensor.def("item", [](const silt::tensor& tensor) -> nb::object {
    if (tensor.elem() != 1)
      throw std::invalid_argument("item(): tensor has more than one element");
    return silt::select(tensor.type(), [&tensor]<silt::primitive S>() -> nb::object {
      if (tensor.host() == silt::CPU)
        return nb::cast(tensor.as<S>()[0]);
      const silt::tensor_t<S> cpu = tensor.as<S>().copy_to(silt::CPU);
      return nb::cast(cpu[0]);
    });
  });

  // Device Switching
  //  `to`/`to_cpu`/`to_gpu` move this tensor's own handle in place (no
  //  new allocation if already on the target host). `copy_to` is the
  //  out-of-place counterpart: it always allocates an independent tensor,
  //  even when the target host matches the source's (see
  //  tensor_t<T>::copy_to(), which both are built on).

  tensor.def("to", [](silt::tensor& tensor, const silt::host_t host) {
    silt::select(tensor.type(), [&tensor, host]<typename T>() {
      tensor.as<T>().to(host);
    });
    return tensor;
  });

  tensor.def("to_cpu", [](silt::tensor& tensor) {
    silt::select(tensor.type(), [&tensor]<typename T>() {
      tensor.as<T>().to_cpu();
    });
    return tensor;
  });

  tensor.def("to_gpu", [](silt::tensor& tensor) {
    silt::select(tensor.type(), [&tensor]<typename T>() {
      tensor.as<T>().to_gpu();
    });
    return tensor;
  });

  tensor.def("copy_to", [](const silt::tensor& tensor, std::optional<silt::host_t> host) {
    const silt::host_t target = host.value_or(tensor.host());
    return silt::select(tensor.type(), [&tensor, target]<silt::primitive S>() -> silt::tensor {
      return silt::tensor(tensor.as<S>().copy_to(target));
    });
  }, nb::arg("host") = nb::none());

  //
  // Tensor Shape Manipulation and Slicing Logic / Tensor Views
  //  Note that these operations are in-place but return copy of self.
  //  We allow for direct subscript, or the explicit "slice" alias.
  //

  tensor.def("reshape", [](silt::tensor& tensor, int d0, int d1, int d2, int d3) {
  tensor.reshape(d0, d1, d2, d3);
  return tensor; }, "d0"_a = 1, "d1"_a = 1, "d2"_a = 1, "d3"_a = 1);

  tensor.def("flatten", [](silt::tensor& tensor) {
    tensor.flatten();
    return tensor;
  });

  tensor.def("__getitem__", silt::detail::getitem_impl);
  tensor.def("slice", silt::detail::getitem_impl);

  //
  // External Library Interop Interface
  //  Note: these conversions COPY the data. Each one allocates a fresh tensor and
  //  copies element-by-element, then hands numpy/pytorch a capsule owning that copy.
  //

  tensor.def("numpy", silt::detail::tensor_to_numpy);

  // So np.asarray(t) works without an explicit .numpy() call. Like
  // .numpy() itself, this is CPU-only -- no implicit host transfer (see
  // the "Explicit Host Placement" design note).
  tensor.def("__array__", [](const silt::tensor& tensor, nb::object, nb::object) {
    return silt::detail::tensor_to_numpy(tensor);
  }, nb::arg("dtype") = nb::none(), nb::arg("copy") = nb::none());

  tensor.def_static("from_numpy", [](const nb::object& object) {
    auto array = nb::cast<nb::ndarray<nb::numpy>>(object);
    if (array.dtype() == nb::dtype<float>()) {
      return silt::detail::tensor_from_numpy<float>(array);
    } else if (array.dtype() == nb::dtype<double>()) {
      return silt::detail::tensor_from_numpy<double>(array);
    } else if (array.dtype() == nb::dtype<int>()) {
      return silt::detail::tensor_from_numpy<int>(array);
    } else {
      throw std::runtime_error("type not supported");
    }
  });

  tensor.def("torch", [](const silt::tensor& tensor) {
    // Unlike numpy(), which is CPU-only by definition, torch tensors can be
    // on either host -- make_torch mirrors source.host() into the
    // returned torch tensor's device, so no host guard is needed here.
    return silt::select(tensor.type(), [&tensor]<typename T>() -> nb::object {
      if constexpr (nb::detail::is_ndarray_scalar_v<T>) {
        return silt::detail::make_torch(tensor.as<T>());
      } else {
        throw std::invalid_argument("tensor type cannot be converted");
      }
    });
  });

  tensor.def_static("from_torch", [](const nb::object& object) {
    auto array = nb::cast<nb::ndarray<nb::pytorch>>(object);
    if (array.dtype() == nb::dtype<float>()) {
      return silt::detail::tensor_from_torch<float>(array);
    } else if (array.dtype() == nb::dtype<double>()) {
      return silt::detail::tensor_from_torch<double>(array);
    } else {
      throw std::runtime_error("type not supported");
    }
  });
}
