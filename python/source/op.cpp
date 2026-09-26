#include <nanobind/nanobind.h>
namespace nb = nanobind;

#include <nanobind/ndarray.h>

#include <nanobind/stl/function.h>
#include <nanobind/stl/string.h>

#include <silt/core/memory.hpp>
#include <silt/core/types.hpp>
#include <silt/core/view.hpp>
#include <silt/op/common.hpp>
#include <silt/op/normal.hpp>

#include "util.hpp"

#include <iostream>

#include "glm.hpp"

template<typename T>
void assert_match(const T& lhs, const T& rhs) {
  if (lhs.type() != rhs.type())
    throw silt::error::mismatch_type(lhs.type(), rhs.type());
  if (lhs.elem() != rhs.elem())
    throw silt::error::mismatch_size(lhs.elem(), rhs.elem());
  if (lhs.host() != rhs.host())
    throw silt::error::mismatch_host(lhs.host(), rhs.host());
}

void bind_op(nb::module_& module) {

  // These bindings mutate their first argument in place, hence the
  // trailing underscore (torch convention). The out-of-place forms
  // (`add`, `multiply`, ...) are a pure-Python wrapper in
  // python/silt/__init__.py: clone the input, then call the `_` form.

  //
  // Binary Operations:
  //  Note that for nanobind, specificity wins in the function parameters.
  //

  module.def("set_", [](silt::tensor& lhs, const silt::tensor& rhs) {
    assert_match(lhs, rhs);
    silt::select(lhs.type(), [&lhs, &rhs]<silt::primitive S>() {
      silt::set<S>(lhs.as<S>(), rhs.as<S>());
    });
  });

  module.def("add_", [](silt::tensor& lhs, const silt::tensor& rhs) {
    assert_match(lhs, rhs);
    silt::select(lhs.type(), [&lhs, &rhs]<silt::primitive S>() {
      silt::add<S>(lhs.as<S>(), rhs.as<S>());
    });
  });

  module.def("multiply_", [](silt::tensor& lhs, const silt::tensor& rhs) {
    assert_match(lhs, rhs);
    silt::select(lhs.type(), [&lhs, &rhs]<silt::primitive S>() {
      silt::multiply<S>(lhs.as<S>(), rhs.as<S>());
    });
  });

  module.def("divide_", [](silt::tensor& lhs, const silt::tensor& rhs) {
    assert_match(lhs, rhs);
    silt::select(lhs.type(), [&lhs, &rhs]<silt::primitive S>() {
      silt::divide<S>(lhs.as<S>(), rhs.as<S>());
    });
  });

  module.def("mix_", [](silt::tensor& lhs, const silt::tensor& rhs, const float w) {
    assert_match(lhs, rhs);
    silt::select(lhs.type(), [&lhs, &rhs, w]<silt::primitive S>() {
      silt::mix<S>(lhs.as<S>(), rhs.as<S>(), w);
    });
  });

  //
  // Unary Operations
  //

  module.def("set_", [](silt::tensor& lhs, const nb::object rhs) {
    silt::select(lhs.type(), [&lhs, &rhs]<silt::primitive S>() {
      silt::set<S>(lhs.as<S>(), nb::cast<S>(rhs));
    });
  });

  module.def("set_", [](silt::view& lhs, const nb::object rhs) {
    silt::select(lhs.type(), [&lhs, &rhs]<silt::primitive S>() {
      silt::set<S>(lhs.as<S>(), nb::cast<S>(rhs));
    });
  });

  module.def("add_", [](silt::tensor& lhs, const nb::object rhs) {
    silt::select(lhs.type(), [&lhs, &rhs]<silt::primitive S>() {
      silt::add<S>(lhs.as<S>(), nb::cast<S>(rhs));
    });
  });

  module.def("add_", [](silt::view& lhs, const nb::object rhs) {
    silt::select(lhs.type(), [&lhs, &rhs]<silt::primitive S>() {
      silt::add<S>(lhs.as<S>(), nb::cast<S>(rhs));
    });
  });

  module.def("multiply_", [](silt::tensor& lhs, const nb::object rhs) {
    silt::select(lhs.type(), [&lhs, &rhs]<silt::primitive S>() {
      silt::multiply<S>(lhs.as<S>(), nb::cast<S>(rhs));
    });
  });

  module.def("multiply_", [](silt::view& lhs, const nb::object rhs) {
    silt::select(lhs.type(), [&lhs, &rhs]<silt::primitive S>() {
      silt::multiply<S>(lhs.as<S>(), nb::cast<S>(rhs));
    });
  });

  module.def("divide_", [](silt::tensor& lhs, const nb::object rhs) {
    silt::select(lhs.type(), [&lhs, &rhs]<silt::primitive S>() {
      silt::divide<S>(lhs.as<S>(), nb::cast<S>(rhs));
    });
  });

  module.def("divide_", [](silt::view& lhs, const nb::object rhs) {
    silt::select(lhs.type(), [&lhs, &rhs]<silt::primitive S>() {
      silt::divide<S>(lhs.as<S>(), nb::cast<S>(rhs));
    });
  });

  module.def("clamp_", [](silt::tensor& lhs, const float min, const float max) {
    silt::select(lhs.type(), [&lhs, min, max]<std::same_as<float> S>() -> void {
      silt::clamp(lhs.as<S>(), min, max);
    });
  });

  module.def("clamp_", [](silt::view& lhs, const float min, const float max) {
    silt::select(lhs.type(), [&lhs, min, max]<std::same_as<float> S>() -> void {
      silt::clamp(lhs.as<S>(), min, max);
    });
  });

  // Tensor Only

  module.def("clone", [](silt::tensor& lhs) {
    return silt::select(lhs.type(), [&lhs]<silt::primitive S>() -> silt::tensor {
      return silt::clone<S>(lhs.as<S>());
    });
  });

  module.def("cast", [](const silt::tensor& tensor, const silt::dtype type) {
    if (tensor.type() == type) {
      return nb::cast(tensor);
    }
    return silt::select(type, [&tensor]<std::floating_point To>() -> nb::object {
      return silt::select(tensor.type(), [&tensor]<std::floating_point From>() -> nb::object {
        silt::tensor result = silt::cast<To, From>(tensor.as<From>());
        return nb::cast(result);
      });
    });
  });

  //
  // Generic Buffer Reductions
  //

  module.def("min", [](const silt::tensor& tensor) {
    return silt::select(tensor.type(), [&tensor]<std::floating_point S>() -> nb::object {
      return nb::cast(silt::min(tensor.as<S>()));
    });
  });

  module.def("max", [](const silt::tensor& tensor) {
    return silt::select(tensor.type(), [&tensor]<std::floating_point S>() -> nb::object {
      return nb::cast(silt::max(tensor.as<S>()));
    });
  });

  //
  // Generic Buffer Functions
  //

  module.def("resize", [](const silt::tensor& rhs, const silt::shape shape) {
    return silt::select(rhs.type(), [&rhs, shape]<silt::primitive S>() -> silt::tensor {
      return silt::tensor(silt::resize<S>(rhs.as<S>(), shape));
    });
  });

  // resample only supports float today (see uncommon.cu) and is a
  // candidate for future deprecation.
  module.def("resample", [](silt::tensor& target, const silt::tensor& source, const silt::vec3 t_scale, const silt::vec3 s_scale, const silt::vec2 pdiff) {
    silt::detail::require_type(target, silt::FLOAT32);
    silt::detail::require_type(source, silt::FLOAT32);
    silt::select(target.type(), [&]<std::same_as<float> S>() {
      silt::resample<S>(target.as<S>(), source.as<S>(), t_scale, s_scale, pdiff);
    });
  });

  //
  // Normal Map ?
  //

  module.def("normal", [](const silt::tensor& tensor, const silt::vec3 scale) {
    if (tensor.host() != silt::CPU)
      throw silt::error::mismatch_host(silt::CPU, tensor.host());

    return silt::select(tensor.type(), [&]<std::floating_point T>() {
      return silt::op::normal(tensor.as<T>(), scale);
    });
  });

  //
  // RNG Operations
  //

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

  //
  // Device Synchronization
  //

  // Blocks until all outstanding GPU work completes, then raises if any
  // kernel launch since the last check left a pending CUDA error (see
  // silt::error::cuda_error and the gpuErrchk calls in operation.hpp).
  module.def("synchronize", &silt::synchronize);
}
