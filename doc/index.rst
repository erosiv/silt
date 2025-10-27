.. silt documentation master file, created by
   sphinx-quickstart on Mon Oct 27 15:42:46 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

silt documentation
==================

simple immediate lightweight tensors

Contents
--------

.. toctree::
   usage
   api_cpp
   api_python
   design

What is silt?
-------------

silt is an isolated lightweight tensor library for easy inclusion in projects that use CUDA with Python bindings. silt is designed for passing around tensor data between various libraries and into kernels on the GPU for physics simulation.

silt is designed to be trivially includable as a git submodule in projects that use a build-system based on ``CMake`` and CUDA (``nvcc``) with python bindings. This enables the designing of **non-monolithic tensor accelerated libraries**.

In essence, silt represents a specific, minimal compilation setup or a kind of `minimal boilerplate glue` that improves build times while keeping interoperability without code duplication.

silt is just over 2000 lines of code (with python bindings), making it extremely legible. In other words, you don't have to use silt, but if you also like to roll your own, then you can at least easily understand its structure and fork it.

Typical Use-Case
----------------

A common use case is to write a small library containing a templated kernel operation:

.. code::

   #include <silt/silt.hpp>
   #include <silt/core/tensor.hpp>

   template<typename T>
   __global__ __kernel(tensor_t<T> tensor);

   template<typename T>
   void my_tensor_operation(silt::tensor_t<T>& tensor) {
      __kernel<<<block, thread>>>(tensor);
   }

Exposed through bindings with ``nanobind``, your library (and all other libraries built with silt) can now operate on silt tensors.

.. code::

   import silt, mylib, otherlib

   shape = silt.shape(1024, 1024)
   tensor = silt.tensor(shape, silt.float32, silt.gpu)
   
   mylib.my_tensor_operation(tensor)
   otherlib.their_tensor_operation(tensor)

Finally, silt takes care of details around memory allocation and deallocation, move and copy semantics, as well as conversion between polymorphic python types and strict-typed C++. silt allows you to no-copy convert tensors on the CPU and GPU to popular libraries like ``numpy`` and ``pytorch``.

Why?
----

I write a lot of tensor accelerated libraries that do fundamentally `different` but `complementary` things. In order to allow them to communicate to build more complex applications, without repeating boilerplate code or introducing heavy dependencies such as pytorch or signifanctly altering my build-system, I isolated the common boilerplate in silt. The libraries can thus remain modular with only a small submodule inclusion.

silt is designed for a specific build system and a specific "scale" - not just the raw memory allocation API, but no unnecessary bells and whistles. This allows it to be trivially included in projects without complicating your build system.