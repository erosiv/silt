#ifndef SILT_PYTHON_LAYER
#define SILT_PYTHON_LAYER

#include <nanobind/nanobind.h>
namespace nb = nanobind;

#include <nanobind/ndarray.h>

#include <nanobind/stl/string.h>
#include <nanobind/stl/function.h>

#include <silt/core/types.hpp>
#include <silt/core/operation.hpp>
#include <silt/op/common.hpp>
#include <silt/op/normal.hpp>

#include <iostream>

#include "glm.hpp"

void bind_op(nb::module_& module) {

//
// Generic Buffer Reductions
//

module.def("min", [](const silt::tensor& tensor){
  return silt::select(tensor.type(), [&tensor]<std::floating_point S>() -> nb::object {
    return nb::cast(silt::min(tensor.as<S>()));
  });
});

module.def("max", [](const silt::tensor& tensor){
  return silt::select(tensor.type(), [&tensor]<std::floating_point S>() -> nb::object {
    return nb::cast(silt::max(tensor.as<S>()));
  });
});

module.def("clamp", [](silt::tensor& tensor, const float min, const float max){
  silt::select(tensor.type(), [&tensor, min, max]<std::same_as<float> S>() -> void {
    silt::clamp(tensor.as<S>(), min, max);
  });
});

//
// Generic Buffer Functions
//

module.def("cast", [](const silt::tensor& tensor, const silt::dtype type){
  if(tensor.type() == type){
    return nb::cast(tensor);
  }
  return silt::select(type, [&tensor]<std::floating_point To>() -> nb::object {
    return silt::select(tensor.type(), [&tensor]<std::floating_point From>() -> nb::object {
      silt::tensor tensor = silt::cast<To, From>(tensor.as<From>());
      return nb::cast(tensor);
    });
  });
});

module.def("copy", [](silt::tensor& lhs, const silt::tensor& rhs, silt::vec2 gmin, silt::vec2 gmax, silt::vec2 gscale, silt::vec2 wmin, silt::vec2 wmax, silt::vec2 wscale, float pscale){

  // Note: This supports copy between different buffer types.
  // The interior template selection just requires that the source
  // buffer's type can be converted to the target buffer's type.

  silt::select(lhs.type(), [&]<silt::primitive To>(){
    silt::select(rhs.type(), [&]<silt::primitive From>(){
      silt::copy<To, From>(lhs.as<To>(), rhs.as<From>(), gmin, gmax, gscale, wmin, wmax, wscale, pscale);
    });
  });
});

module.def("resize", [](const silt::tensor& rhs, const silt::shape shape){
  return silt::select(rhs.type(), [&rhs, shape]<silt::primitive S>() -> silt::tensor {
    return silt::tensor(silt::resize<S>(rhs.as<S>(), shape));
  });
});

module.def("resample", [](silt::tensor& target, const silt::tensor& source, const silt::vec3 t_scale, const silt::vec3 s_scale, const silt::vec2 pdiff){
  silt::select(target.type(), [&]<silt::primitive S>() {
    silt::resample<S>(target.as<S>(), source.as<S>(), t_scale, s_scale, pdiff);
  });
});

module.def("set", [](silt::tensor& lhs, const silt::tensor& rhs){

  if(lhs.type() != rhs.type())
    throw silt::error::mismatch_type(lhs.type(), rhs.type());
  
  if (lhs.elem() != rhs.elem())
    throw silt::error::mismatch_size(lhs.elem(), rhs.elem());

  if (lhs.host() != rhs.host())
    throw silt::error::mismatch_host(lhs.host(), rhs.host());
  
  silt::select(lhs.type(), [&lhs, &rhs]<silt::primitive S>(){
    silt::set<S>(lhs.as<S>(), rhs.as<S>());
  });
});

module.def("set", [](silt::tensor& tensor, const nb::object value){
  silt::select(tensor.type(), [&tensor, &value]<silt::primitive S>(){
    auto tensor_t = tensor.as<S>();
    auto value_t = nb::cast<S>(value);
    silt::set<S>(tensor_t, value_t);
  });
});

module.def("add", [](silt::tensor& lhs, const silt::tensor& rhs){

  if(lhs.type() != rhs.type())
    throw silt::error::mismatch_type(lhs.type(), rhs.type());

  if (lhs.elem() != rhs.elem())
    throw silt::error::mismatch_size(lhs.elem(), rhs.elem());

  if (lhs.host() != rhs.host())
    throw silt::error::mismatch_host(lhs.host(), rhs.host());

  silt::select(lhs.type(), [&lhs, &rhs]<silt::primitive S>(){
    silt::add<S>(lhs.as<S>(), rhs.as<S>());
  });
});

module.def("add", [](silt::tensor& buffer, const nb::object value){
  silt::select(buffer.type(), [&buffer, &value]<silt::primitive S>(){
    auto buffer_t = buffer.as<S>();
    auto value_t = nb::cast<S>(value);
    silt::add<S>(buffer_t, value_t);
  });
});

module.def("multiply", [](silt::tensor& lhs, const silt::tensor& rhs){
  
  if(lhs.type() != rhs.type())
    throw silt::error::mismatch_type(lhs.type(), rhs.type());

  if (lhs.elem() != rhs.elem())
    throw silt::error::mismatch_size(lhs.elem(), rhs.elem());

  if (lhs.host() != rhs.host())
    throw silt::error::mismatch_host(lhs.host(), rhs.host());

  silt::select(lhs.type(), [&lhs, &rhs]<silt::primitive S>(){
    silt::multiply<S>(lhs.as<S>(), rhs.as<S>());
  });
});

module.def("multiply", [](silt::tensor& buffer, const nb::object value){
  silt::select(buffer.type(), [&buffer, &value]<silt::primitive S>(){
    auto buffer_t = buffer.as<S>();
    auto value_t = nb::cast<S>(value);
    silt::multiply<S>(buffer_t, value_t);
  });
});

//
// Normal Map ?
//

module.def("normal", [](const silt::tensor& tensor, const silt::vec3 scale){

  if (tensor.host() != silt::CPU)
    throw silt::error::mismatch_host(silt::CPU, tensor.host());

  return silt::select(tensor.type(), [&]<std::floating_point T>(){
    return silt::op::normal(tensor.as<T>(), scale);
  });

});

//
// RNG Operations
//

module.def("seed", [](silt::tensor& tensor, const size_t seed, const size_t offset){
  return silt::seed(tensor.as<silt::rng>(), seed, offset);
});

module.def("sample_uniform", [](silt::tensor& tensor){
  return silt::tensor(silt::sample_uniform(tensor.as<silt::rng>()));
});

module.def("sample_uniform", [](silt::tensor& tensor, const float min, const float max){
  return silt::tensor(silt::sample_uniform(tensor.as<silt::rng>(), min, max));
});

module.def("sample_normal", [](silt::tensor& tensor){
  return silt::tensor(silt::sample_normal(tensor.as<silt::rng>()));
});

module.def("sample_normal", [](silt::tensor& tensor, const float mean, const float std){
  return silt::tensor(silt::sample_normal(tensor.as<silt::rng>(), mean, std));
});

}

#endif