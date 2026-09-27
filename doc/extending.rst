Extending silt
==============

This page describes the contract for writing a library that operates on silt
tensors. It is the counterpart to :doc:`design`, which explains *how* the
polymorphic layer works; this page explains *what you have to do* to plug into it.

The short version: write strict-typed templated kernels against
``silt::tensor_t<T>``, explicitly instantiate them for the types you support,
mark the instantiations ``EXPORT_SHARED``, and bind them with a ``silt::select``
call that recovers the static type from the runtime tag.

Linking against silt
--------------------

silt builds two targets. ``silt_lib`` (aliased as ``silt::silt``) is the shared
C++/CUDA library; ``silt`` is the Python module. As a submodule consumer you
want the former:

.. code::
  CMake

  add_subdirectory(${CMAKE_SOURCE_DIR}/ext/silt silt)
  target_link_libraries(${TARGET_NAME} PUBLIC silt_lib)

Linking ``silt_lib`` is what makes your library interoperable with every other
library built against the same silt: a ``silt::tensor`` created by one can be
passed to another, because both resolve the same exported symbols and the same
polymorphic implementation pointer.

Note that silt deliberately hides the CUDA runtime behind its own interface
(``silt/core/memory.hpp``), so only silt itself links against the CUDA runtime.
Your library does not need to.

Symbol export rules
-------------------

On Windows, everything reachable across the DLL boundary must be explicitly
exported, and this includes free functions -- not just types. ``EXPORT_SHARED``
(defined in ``silt/silt.hpp``) expands to ``__declspec(dllexport)`` or
``dllimport`` as appropriate, and to a visibility attribute elsewhere.

The rules, which are easy to get subtly wrong:

- **Generic templates are not** ``EXPORT_SHARED``. A template is not a symbol;
  there is nothing to export until it is instantiated.
- **Explicit template instantiations are** ``EXPORT_SHARED``. The instantiation
  is what produces the specialized symbol, so that is where the attribute goes.
- **Specialized templates in headers are** ``EXPORT_SHARED``, including bare
  declarations.
- **Non-template functions in headers are** ``EXPORT_SHARED``, including bare
  declarations.

Concretely, from ``silt/op/common_unary.cu``:

.. code::
  C++

  // The generic template: no EXPORT_SHARED.
  template<typename T>
  void set(tensor_t<T> lhs, const T rhs) {
    op::uniop_inplace(lhs, [rhs] GPU_ENABLE (const T a){ return rhs; });
  }

  // The instantiations: EXPORT_SHARED goes here.
  template EXPORT_SHARED void silt::set<int>   (silt::tensor_t<int> lhs,    const int rhs);
  template EXPORT_SHARED void silt::set<float> (silt::tensor_t<float> lhs,  const float rhs);
  template EXPORT_SHARED void silt::set<double>(silt::tensor_t<double> lhs, const double rhs);

Explicit instantiation is required because the kernel bodies live in ``.cu``
translation units compiled by ``nvcc``, while the call sites live in ``.cpp``
translation units compiled by the host compiler. The host compiler never sees
the template body, so the symbol has to exist already.

Host and device functions
-------------------------

``GPU_ENABLE`` marks a function as callable from both host and device. It expands
to ``__host__ __device__`` when ``HAS_CUDA`` is defined and to nothing otherwise,
which is what lets the same ``tensor_t``, ``shape``, ``slice`` and ``view_t``
headers be consumed by both ``nvcc`` and the host compiler.

``HAS_CUDA`` must be defined consistently for every translation unit that
includes silt headers and is compiled by ``nvcc``. Defining it for some
translation units and not others means the inline members of those shared types
have different definitions in different units, which is an ODR violation --
benign in practice today, but not something to rely on.

Writing a kernel operation
--------------------------

The typical shape of a downstream operation:

.. code::
  C++

  // mylib/erode.cuh
  #include <silt/core/tensor.hpp>

  namespace mylib {

  template<typename T>
  void erode(silt::tensor_t<T> height, const T rate);

  }

.. code::
  C++

  // mylib/erode.cu
  #define HAS_CUDA
  #include "erode.cuh"

  namespace mylib {

  template<typename T>
  __global__ void __erode(silt::tensor_t<T> height, const T rate) {
    const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= height.elem()) return;
    // ... operate on height[n] ...
  }

  template<typename T>
  void erode(silt::tensor_t<T> height, const T rate) {
    __erode<<<(height.elem() + 511) / 512, 512>>>(height, rate);
  }

  template EXPORT_SHARED void mylib::erode<float> (silt::tensor_t<float> h,  const float r);
  template EXPORT_SHARED void mylib::erode<double>(silt::tensor_t<double> h, const double r);

  }

Note that ``tensor_t<T>`` is passed **by value** into kernels. It is reference
counted, so this is cheap, and the copy is what makes the raw data pointer
available on the device.

Generic elementwise operations
------------------------------

Before writing a kernel by hand, check whether ``silt/core/operation.hpp``
already covers it. It provides host/device-dispatching map and zip primitives
that accept an arbitrary functor, so most elementwise work needs no kernel of
your own:

.. code::
  C++

  #include <silt/core/operation.hpp>

  // unary, in place: lhs[n] = func(lhs[n])
  silt::op::uniop_inplace(tensor, [] GPU_ENABLE (const float a){ return a * 0.5f; });

  // binary, in place: lhs[n] = func(lhs[n], rhs[n])
  silt::op::binop_inplace(lhs, rhs, [] GPU_ENABLE (const float a, const float b){
    return a + b;
  });

These dispatch on ``lhs.host()`` and run either a host loop or a CUDA kernel.
They accept both ``tensor_t<T>`` and ``view_t<T>``, so the same functor works on
whole tensors and on slices.

Two caveats. These primitives do **not** validate that ``lhs`` and ``rhs`` agree
in shape or live on the same device -- that is the caller's responsibility. And
because they take an arbitrary functor, the lambda must be marked ``GPU_ENABLE``
and the translation unit must be compiled with ``--extended-lambda``.

Exposing your operation to Python
---------------------------------

Bind the polymorphic wrapper, not the strict-typed template, and use
``silt::select`` to recover the static type from the runtime tag. Constrain the
lambda's template parameter to the types your kernel actually supports -- a
mismatched type then raises at runtime instead of failing to compile:

.. code::
  C++

  module.def("erode", [](silt::tensor& height, const float rate){
    silt::select(height.type(), [&]<std::floating_point S>(){
      mylib::erode<S>(height.as<S>(), S(rate));
    });
  });

``as<T>()`` is an unchecked ``static_cast`` on the implementation pointer. Inside
a ``select`` the tag has just been switched on, so the cast is sound. Calling
``as<T>()`` **outside** a ``select``, on a tag you have not verified, is type
confusion -- check ``type()`` first.

See :doc:`design` for the details of the ``select`` mechanism and how concept
constraints on the lambda parameter restrict which instantiations are generated.
