#include <nanobind/nanobind.h>
namespace nb = nanobind;

#include <nanobind/ndarray.h>

#include <nanobind/stl/function.h>
#include <nanobind/stl/string.h>

#include <silt/core/memory.hpp>
#include <silt/core/types.hpp>
#include <silt/core/view.hpp>
#include <silt/op/common.hpp>
#include <silt/op/reduce.hpp>

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
  // python/silt/__init__.py: copy_to() the input, then call the `_` form.

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

  module.def("set_", [](silt::view& lhs, const silt::view& rhs) {
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

  module.def("add_", [](silt::view& lhs, const silt::view& rhs) {
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

  module.def("multiply_", [](silt::view& lhs, const silt::view& rhs) {
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

  module.def("divide_", [](silt::view& lhs, const silt::view& rhs) {
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

  module.def("mix_", [](silt::view& lhs, const silt::view& rhs, const float w) {
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

  module.def("minimum_", [](silt::tensor& lhs, const silt::tensor& rhs) {
    assert_match(lhs, rhs);
    silt::select(lhs.type(), [&lhs, &rhs]<silt::primitive S>() {
      silt::minimum<S>(lhs.as<S>(), rhs.as<S>());
    });
  });

  module.def("minimum_", [](silt::view& lhs, const silt::view& rhs) {
    assert_match(lhs, rhs);
    silt::select(lhs.type(), [&lhs, &rhs]<silt::primitive S>() {
      silt::minimum<S>(lhs.as<S>(), rhs.as<S>());
    });
  });

  module.def("minimum_", [](silt::tensor& lhs, const nb::object rhs) {
    silt::select(lhs.type(), [&lhs, &rhs]<silt::primitive S>() {
      silt::minimum<S>(lhs.as<S>(), nb::cast<S>(rhs));
    });
  });

  module.def("minimum_", [](silt::view& lhs, const nb::object rhs) {
    silt::select(lhs.type(), [&lhs, &rhs]<silt::primitive S>() {
      silt::minimum<S>(lhs.as<S>(), nb::cast<S>(rhs));
    });
  });

  module.def("maximum_", [](silt::tensor& lhs, const silt::tensor& rhs) {
    assert_match(lhs, rhs);
    silt::select(lhs.type(), [&lhs, &rhs]<silt::primitive S>() {
      silt::maximum<S>(lhs.as<S>(), rhs.as<S>());
    });
  });

  module.def("maximum_", [](silt::view& lhs, const silt::view& rhs) {
    assert_match(lhs, rhs);
    silt::select(lhs.type(), [&lhs, &rhs]<silt::primitive S>() {
      silt::maximum<S>(lhs.as<S>(), rhs.as<S>());
    });
  });

  module.def("maximum_", [](silt::tensor& lhs, const nb::object rhs) {
    silt::select(lhs.type(), [&lhs, &rhs]<silt::primitive S>() {
      silt::maximum<S>(lhs.as<S>(), nb::cast<S>(rhs));
    });
  });

  module.def("maximum_", [](silt::view& lhs, const nb::object rhs) {
    silt::select(lhs.type(), [&lhs, &rhs]<silt::primitive S>() {
      silt::maximum<S>(lhs.as<S>(), nb::cast<S>(rhs));
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
  // Dense Reductions
  //

  module.def("sum", [](const silt::tensor& tensor) {
    return silt::select(tensor.type(), [&tensor]<silt::primitive S>() -> nb::object {
      return nb::cast(silt::sum(tensor.as<S>()));
    });
  });

  module.def("mean", [](const silt::tensor& tensor) {
    return silt::select(tensor.type(), [&tensor]<silt::primitive S>() -> nb::object {
      return nb::cast(silt::mean(tensor.as<S>()));
    });
  });

  module.def("min", [](const silt::tensor& tensor) {
    return silt::select(tensor.type(), [&tensor]<silt::primitive S>() -> nb::object {
      return nb::cast(silt::min(tensor.as<S>()));
    });
  });

  module.def("max", [](const silt::tensor& tensor) {
    return silt::select(tensor.type(), [&tensor]<silt::primitive S>() -> nb::object {
      return nb::cast(silt::max(tensor.as<S>()));
    });
  });

  module.def("var", [](const silt::tensor& tensor) {
    return silt::select(tensor.type(), [&tensor]<silt::primitive S>() -> nb::object {
      return nb::cast(silt::variance(tensor.as<S>()));
    });
  });

  module.def("std", [](const silt::tensor& tensor) {
    return silt::select(tensor.type(), [&tensor]<silt::primitive S>() -> nb::object {
      return nb::cast(silt::stddev(tensor.as<S>()));
    });
  });

  module.def("argmin", [](const silt::tensor& tensor) {
    return silt::select(tensor.type(), [&tensor]<silt::primitive S>() -> nb::object {
      return nb::cast(silt::argmin(tensor.as<S>()));
    });
  });

  module.def("argmax", [](const silt::tensor& tensor) {
    return silt::select(tensor.type(), [&tensor]<silt::primitive S>() -> nb::object {
      return nb::cast(silt::argmax(tensor.as<S>()));
    });
  });

  //
  // Device Synchronization
  //

  // Blocks until all outstanding GPU work completes, then raises if any
  // kernel launch since the last check left a pending CUDA error (see
  // silt::error::cuda_error and the gpuErrchk calls in operation.hpp).
  module.def("synchronize", &silt::synchronize);
}
