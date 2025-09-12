#ifndef SILT_PYTHON_UTIL
#define SILT_PYTHON_UTIL

#include <nanobind/nanobind.h>
namespace nb = nanobind;

#include <nanobind/ndarray.h>
#include <silt/core/tensor.hpp>
#include <silt/op/common.hpp>
#include "interop.hpp"

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