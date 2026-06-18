#ifndef SILT_PYTHON_UTIL
#define SILT_PYTHON_UTIL

#include <nanobind/nanobind.h>
namespace nb = nanobind;
using namespace nb::literals;

#include <nanobind/ndarray.h>
#include <silt/core/tensor.hpp>
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

    // Slice the View:
    for(size_t d = 0; d < size; ++d) {

      nb::handle handle = tuple[d];
      Py_ssize_t offset, stride, extent;
      __unpack_slice(handle, offset, stride, extent);
      extent = std::min((shape.ext()[d] - offset) / stride, extent);
      view_t.index(d, offset, stride, extent);

    }

    return silt::view(view_t);
  
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
//  Note: Memory is shared, not copied.
//  The lifetimes of the objects are managed
//  so that the memory is not deleted.
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
  if(tensor.host() != silt::host_t::GPU)
    throw silt::error::unsupported_host(silt::host_t::GPU, tensor.host());
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

#endif