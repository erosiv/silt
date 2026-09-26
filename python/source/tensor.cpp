#include <nanobind/nanobind.h>
namespace nb = nanobind;
using namespace nb::literals;

#include <nanobind/ndarray.h>
#include <silt/core/tensor.hpp>
#include <silt/core/view.hpp>
#include <silt/op/common.hpp>
#include "interop.hpp"
#include "util.hpp"

silt::view __slice(silt::tensor& tensor, nb::tuple tuple) {
  return silt::select(tensor.type(), [&tensor, tuple]<typename T>() -> silt::view {

    // Construct a View from the Tensor:
    auto tensor_t = tensor.as<T>();
    auto view_t = tensor_t.template view<T>();
    auto shape = tensor_t.shape();
    // by default, the view adopts the tensor's shape
    view_t.reshape(shape[0], shape[1], shape[2], shape[3]);

    // Validate Number of Dimensions
    const size_t size = tuple.size();
    if(size != shape.dim()) {
      throw silt::error::mismatch_size(shape.dim(), size);
    }

    // Slice the View: unpack raw offset/stride/extent and forward to
    // view_t::index, which forwards to slice::index -- the single place
    // slicing is clamped and validated (see slice.hpp).
    for(size_t d = 0; d < size; ++d) {

      nb::handle handle = tuple[d];
      Py_ssize_t offset, stride, extent;
      __unpack_slice(handle, offset, stride, extent);
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

//! General Util Binding Function
void bind_tensor(nb::module_& module){

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

// Device Switching

tensor.def("cpu", [](silt::tensor& tensor){
  silt::select(tensor.type(), [&tensor]<typename T>(){
    tensor.as<T>().to_cpu();
  });
  return tensor;
});

tensor.def("gpu", [](silt::tensor& tensor){
  silt::select(tensor.type(), [&tensor]<typename T>(){
    tensor.as<T>().to_gpu();
  });
  return tensor;
});

//
// Tensor Shape Manipulation and Slicing Logic / Tensor Views
//  Note that these operations are in-place but return copy of self.
//  We allow for direct subscript, or the explicit "slice" alias.
//

tensor.def("reshape", [](silt::tensor& tensor, int d0, int d1, int d2, int d3) {
  tensor.reshape(d0, d1, d2, d3);
  return tensor;
}, "d0"_a = 1, "d1"_a = 1, "d2"_a = 1, "d3"_a = 1);

tensor.def("flatten", [](silt::tensor& tensor) {
  tensor.flatten();
  return tensor;
});

tensor.def("__getitem__", __slice);
tensor.def("slice", __slice);

//
// External Library Interop Interface
//  Note: these conversions COPY the data. Each one allocates a fresh tensor and
//  copies element-by-element, then hands numpy/pytorch a capsule owning that copy.
//  tensor_t is already reference counted, so a genuinely zero-copy path is possible
//  by making the capsule own a refcount-incremented handle to the source instead.
//  See the 1.2 plan (E4) -- the docs previously claimed the no-copy behaviour.
//

tensor.def("numpy", [](const silt::tensor& tensor){
  if(tensor.host() != silt::host_t::CPU)
    throw silt::error::unsupported_host(silt::host_t::CPU, tensor.host());
  return silt::select(tensor.type(), [&tensor]<typename T>() -> nb::object {
    if constexpr(nb::detail::is_ndarray_scalar_v<T>){
      return __make_numpy(tensor.as<T>());
    } else {
      throw std::invalid_argument("tensor type cannot be converted");
    }
  });
});

tensor.def_static("from_numpy", [](const nb::object& object){
  auto array = nb::cast<nb::ndarray<nb::numpy>>(object);
  if(array.dtype() == nb::dtype<float>()){
    return __tensor_from_numpy<float>(array);
  } else if(array.dtype() == nb::dtype<double>()){
    return __tensor_from_numpy<double>(array);
  } else if(array.dtype() == nb::dtype<int>()){
    return __tensor_from_numpy<int>(array);
  } else {
    throw std::runtime_error("type not supported");
  }
});

tensor.def("torch", [](const silt::tensor& tensor){
  // Unlike numpy(), which is CPU-only by definition, torch tensors can be
  // on either host -- __make_torch mirrors source.host() into the
  // returned torch tensor's device, so no host guard is needed here.
  return silt::select(tensor.type(), [&tensor]<typename T>() -> nb::object {
    if constexpr(nb::detail::is_ndarray_scalar_v<T>){
      return __make_torch(tensor.as<T>());
    } else {
      throw std::invalid_argument("tensor type cannot be converted");
    }
  });
});

tensor.def_static("from_torch", [](const nb::object& object){
  auto array = nb::cast<nb::ndarray<nb::pytorch>>(object);
  if(array.dtype() == nb::dtype<float>()){
    return __tensor_from_torch<float>(array);
  } else if(array.dtype() == nb::dtype<double>()){
    return __tensor_from_torch<double>(array);
  } else {
    throw std::runtime_error("type not supported");
  }
});

}
