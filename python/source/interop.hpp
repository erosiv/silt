#pragma once

#include <cstring>
#include <stdexcept>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <silt/core/tensor.hpp>
#include <silt/op/common.hpp>

namespace nb = nanobind;

namespace silt {
namespace detail {

// Everything here is a helper behind tensor.cpp's numpy()/from_numpy()/
// torch()/from_torch() bindings, not part of the public API, and lives
// in `detail` instead of using a double-underscore name (which is
// reserved to the implementation at any scope, namespace or not).

//
// Numpy Buffer from Type Buffer Generator
//

template<typename T, size_t D>
nb::object make_numpy(T* data, const silt::shape shape, nb::capsule owner) {

  size_t _shape[D]{0};
  for (size_t d = 0; d < D; ++d)
    _shape[d] = shape[d];

  nb::ndarray<nb::numpy, T, nb::ndim<D>> array(
      data,
      D,
      _shape,
      owner
  );
  return nb::cast(std::move(array));
}

template<typename T>
nb::object make_numpy(const silt::tensor_t<T>& source) {

  const silt::shape shape = source.shape();
  silt::tensor_t<T>* target = new silt::tensor_t<T>(shape, source.host());
  nb::capsule owner(target, [](void* p) noexcept {
    delete (silt::tensor_t<T>*)p;
  });
  silt::set(*target, source);

  // note: if the object comes in as a python object, we can tie the lifetime
  //  of the original object to the existence of the numpy object if the
  //  memory is shared and not copied. Note that here, we are copying!!!
  // owner = nb::find(silt::tensor source) // <- use find operator on
  //  object with original python pointer.

  switch (shape.dim()) {
  case 0: // a single-element tensor is treated as rank-1
  case 1:
    return make_numpy<T, 1>(target->data(), shape, owner);
  case 2:
    return make_numpy<T, 2>(target->data(), shape, owner);
  case 3:
    return make_numpy<T, 3>(target->data(), shape, owner);
  case 4:
    return make_numpy<T, 4>(target->data(), shape, owner);
  default:
    throw std::invalid_argument("too many dimensions");
  }
}

//! An array is C-contiguous (ignoring any size-1 dimension, whose stride
//! numpy/torch leave unspecified) iff each dimension's stride equals the
//! product of the extents to its right -- i.e. walking dimensions from
//! the last to the first, the expected stride for a flat scan. Templated
//! on the array type so it works for both nb::ndarray<nb::numpy> and
//! nb::ndarray<nb::pytorch> -- shape()/stride()/ndim() are the same API
//! on both.
template<typename Array>
inline bool is_c_contiguous(const Array& array) {

  int64_t expected = 1;
  for (size_t i = array.ndim(); i-- > 0;) {
    if (array.shape(i) != 1 && array.stride(i) != expected)
      return false;
    expected *= (int64_t)array.shape(i);
  }
  return true;
}

template<typename T>
silt::tensor tensor_from_numpy(const nb::ndarray<nb::numpy>& array) {

  const size_t ndim = array.ndim();
  const int d0 = (ndim >= 1) ? array.shape(0) : 1;
  const int d1 = (ndim >= 2) ? array.shape(1) : 1;
  const int d2 = (ndim >= 3) ? array.shape(2) : 1;
  const int d3 = (ndim >= 4) ? array.shape(3) : 1;
  auto shape = silt::shape(d0, d1, d2, d3);

  auto tensor_t = silt::tensor_t<T>(shape, silt::host_t::CPU);
  const T* data = (const T*)array.data();

  if (is_c_contiguous(array)) {

    // Fast path: numpy's buffer is already laid out exactly the way silt
    // wants it, so a flat memcpy is correct. The source array is only
    // read here, never written to.
    std::memcpy(tensor_t.data(), data, tensor_t.size());

  } else {

    // Slow path: a transposed or otherwise strided array. Read through
    // numpy's own per-dimension strides (in elements, not bytes -- see
    // nb::ndarray::stride()) instead of scanning the backing buffer as
    // though it were flat. Still read-only on the source array.
    const int64_t s0 = (ndim >= 1) ? array.stride(0) : 0;
    const int64_t s1 = (ndim >= 2) ? array.stride(1) : 0;
    const int64_t s2 = (ndim >= 3) ? array.stride(2) : 0;
    const int64_t s3 = (ndim >= 4) ? array.stride(3) : 0;

    int64_t flat = 0;
    for (int i0 = 0; i0 < d0; ++i0)
      for (int i1 = 0; i1 < d1; ++i1)
        for (int i2 = 0; i2 < d2; ++i2)
          for (int i3 = 0; i3 < d3; ++i3)
            tensor_t[flat++] = data[i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3];
  }

  return std::move(silt::tensor(tensor_t));
}

//
// PyTorch Tensor Generation
//

template<typename T, size_t D>
nb::object make_torch(T* data, const silt::shape shape, nb::capsule owner, int device_type) {

  size_t _shape[D]{0};
  for (size_t d = 0; d < D; ++d)
    _shape[d] = shape[d];

  nb::ndarray<nb::pytorch, T, nb::ndim<D>> array(
      data,
      D,
      _shape,
      owner,
      nullptr,
      nb::dtype<T>(),
      device_type
  );
  return nb::cast(std::move(array));
}

template<typename T>
nb::object make_torch(const silt::tensor_t<T>& source) {

  const silt::shape shape = source.shape();
  silt::tensor_t<T>* target = new silt::tensor_t<T>(shape, source.host());
  nb::capsule owner(target, [](void* p) noexcept {
    delete (silt::tensor_t<T>*)p;
  });
  silt::set(*target, source);

  // A CPU-hosted silt tensor becomes a CPU torch tensor, a GPU-hosted
  // one a CUDA torch tensor -- target was just allocated on source's
  // own host above, so this always matches what target->data() points at.
  const int device_type = (source.host() == silt::host_t::GPU)
                              ? nb::device::cuda::value
                              : nb::device::cpu::value;

  switch (shape.dim()) {
  case 0: // a single-element tensor is treated as rank-1
  case 1:
    return make_torch<T, 1>(target->data(), shape, owner, device_type);
  case 2:
    return make_torch<T, 2>(target->data(), shape, owner, device_type);
  case 3:
    return make_torch<T, 3>(target->data(), shape, owner, device_type);
  case 4:
    return make_torch<T, 4>(target->data(), shape, owner, device_type);
  default:
    throw std::invalid_argument("too many dimensions");
  }
}

template<typename T>
silt::tensor tensor_from_torch(const nb::ndarray<nb::pytorch>& array) {

  const size_t ndim = array.ndim();
  const int d0 = (ndim >= 1) ? array.shape(0) : 1;
  const int d1 = (ndim >= 2) ? array.shape(1) : 1;
  const int d2 = (ndim >= 3) ? array.shape(2) : 1;
  const int d3 = (ndim >= 4) ? array.shape(3) : 1;
  auto shape = silt::shape(d0, d1, d2, d3);

  if (array.device_type() == nb::device::cpu::value) {

    // CPU-resident torch tensor: same story as numpy -- contiguity is
    // not guaranteed (a .t() or a slice), so mirror
    // tensor_from_numpy's fast/slow path exactly.
    auto tensor_t = silt::tensor_t<T>(shape, silt::host_t::CPU);
    const T* data = (const T*)array.data();

    if (is_c_contiguous(array)) {

      std::memcpy(tensor_t.data(), data, tensor_t.size());

    } else {

      const int64_t s0 = (ndim >= 1) ? array.stride(0) : 0;
      const int64_t s1 = (ndim >= 2) ? array.stride(1) : 0;
      const int64_t s2 = (ndim >= 3) ? array.stride(2) : 0;
      const int64_t s3 = (ndim >= 4) ? array.stride(3) : 0;

      int64_t flat = 0;
      for (int i0 = 0; i0 < d0; ++i0)
        for (int i1 = 0; i1 < d1; ++i1)
          for (int i2 = 0; i2 < d2; ++i2)
            for (int i3 = 0; i3 < d3; ++i3)
              tensor_t[flat++] = data[i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3];
    }

    return std::move(silt::tensor(tensor_t));

  } else if (array.device_type() == nb::device::cuda::value) {

    // GPU-resident torch tensor. silt has no strided GPU copy (device_copy
    // and the kernels behind silt::set assume contiguous storage), so a
    // non-contiguous CUDA tensor is rejected rather than silently
    // misread -- call .contiguous() on the torch side first.
    if (!is_c_contiguous(array))
      throw std::invalid_argument("from_torch: non-contiguous CUDA tensor is not supported, call .contiguous() first");

    T* data = (T*)array.data();
    auto target_t = silt::tensor_t<T>(shape, silt::host_t::GPU);
    silt::set(target_t, silt::tensor_t<T>(data, shape, silt::host_t::GPU));
    return std::move(silt::tensor(target_t));

  } else {

    // Previously this branch didn't exist at all: any non-CUDA tensor
    // (including a perfectly ordinary CPU tensor) had its pointer handed
    // straight to a GPU kernel.
    throw std::invalid_argument("from_torch: tensor must be on cpu or cuda");
  }
}

} // namespace detail
} // namespace silt