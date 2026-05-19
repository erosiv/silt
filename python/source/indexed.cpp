#ifndef SILT_PYTHON_INDEXED
#define SILT_PYTHON_INDEXED

#include <nanobind/nanobind.h>
namespace nb = nanobind;

#include <silt/op/indexed.hpp>
#include "glm.hpp"

void bind_indexed(nb::module_& module) {

module.def("indexed_set", [](silt::tensor& lhs, const nb::object value, const silt::tensor& ind) {

  if(lhs.host() != ind.host())
    throw silt::error::mismatch_host(lhs.host(), ind.host());
  
  if(ind.type() != silt::dtype::INT32)
    throw silt::error::mismatch_type(lhs.type(), silt::dtype::INT32);

  silt::select(lhs.type(), [&lhs, &value, &ind]<silt::primitive S>(){
    auto tensor_t = lhs.as<S>();
    auto value_t = nb::cast<S>(value);
    auto index_t = ind.as<int>();
    silt::indexed_set<S>(tensor_t, value_t, index_t);
  });

});

module.def("index_radius", [](const silt::shape& shape, const silt::vec2 center, const float radius){
  return silt::tensor(silt::index_radius(shape, center, radius));
});

}

#endif