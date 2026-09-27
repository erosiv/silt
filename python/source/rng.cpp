#include <nanobind/nanobind.h>
namespace nb = nanobind;

#include <silt/op/rng.hpp>

#include "util.hpp"

void bind_rng(nb::module_& module) {

  module.def("seed", [](silt::tensor& tensor, const size_t seed, const size_t offset) {
    silt::detail::require_type(tensor, silt::RNG);
    return silt::seed(tensor.as<silt::rng>(), seed, offset);
  });

  module.def("sample_uniform", [](silt::tensor& tensor) {
    silt::detail::require_type(tensor, silt::RNG);
    return silt::tensor(silt::sample_uniform(tensor.as<silt::rng>()));
  });

  module.def("sample_uniform", [](silt::tensor& tensor, const float min, const float max) {
    silt::detail::require_type(tensor, silt::RNG);
    return silt::tensor(silt::sample_uniform(tensor.as<silt::rng>(), min, max));
  });

  module.def("sample_normal", [](silt::tensor& tensor) {
    silt::detail::require_type(tensor, silt::RNG);
    return silt::tensor(silt::sample_normal(tensor.as<silt::rng>()));
  });

  module.def("sample_normal", [](silt::tensor& tensor, const float mean, const float std) {
    silt::detail::require_type(tensor, silt::RNG);
    return silt::tensor(silt::sample_normal(tensor.as<silt::rng>(), mean, std));
  });
}
