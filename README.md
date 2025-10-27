# silt

simple immediate lightweight tensors

## What is silt?

silt is an isolated lightweight tensor library for easy inclusion in projects that use CUDA with Python bindings. silt is designed for passing around tensor data between various libraries and into kernels on the GPU for physics simulation.

silt is designed to be trivially includable as a git submodule in projects that use a build-system based on ``CMake`` and CUDA (``nvcc``) with python bindings. This enables the designing of **non-monolithic tensor accelerated libraries**.

In essence, silt represents a specific, minimal compilation setup or a kind of `minimal boilerplate glue` that improves build times while keeping interoperability without code duplication.

silt is just over 2000 lines of code (with python bindings), making it extremely legible. In other words, you don't have to use silt, but if you also like to roll your own, then you can at least easily understand its structure and fork it.

## Features

- `silt` supports both CPU and GPU tensors based on CUDA.
- `silt` is interoperable with python numpy and pytorch tensors.
- `silt` provides a basic set of common immediate-evaluation tensor operations.
- `silt` supports 1-4 dimensional tensors, intended for physics simulations.
- `silt` is fully statically compiled in C++/CUDA while polymorphic in the python interface

The goal is to be super lightweight with minimal compile times and easy inclusion into CMake projects via git submodules.

Note that this library doesn't implement complicated operations or features like autodifferentiation to provide a clean interface and minimal implementation.

## Usage

### Install Python Module

*coming soon to PyPI.org*

### Typical Use-Case

A common use case is to write a small library containing a templated kernel operation:

```c++
#include <silt/silt.hpp>
#include <silt/core/tensor.hpp>

template<typename T>
__global__ __kernel(tensor_t<T> tensor);

template<typename T>
void my_tensor_operation(silt::tensor_t<T>& tensor) {
  __kernel<<<block, thread>>>(tensor);
}
```

Exposed through bindings with ``nanobind``, your library (and all other libraries built with silt) can now operate on silt tensors.

```python
import silt, mylib, otherlib

shape = silt.shape(1024, 1024)
tensor = silt.tensor(shape, silt.float32, silt.gpu)

mylib.my_tensor_operation(tensor)
otherlib.their_tensor_operation(tensor)
```

Finally, silt takes care of details around memory allocation and deallocation, move and copy semantics, as well as conversion between polymorphic python types and strict-typed C++. silt allows you to no-copy convert tensors on the CPU and GPU to popular libraries like ``numpy`` and ``pytorch``.

## Build from Scratch

### Build Python Module

Install `silt` using `pip`:

```bash
pip install .
```

No build-isolation progressive build (development):

```bash
pip install --no-build-isolation -ve .
```

Build a distributable `.whl` file:

```bash
pip wheel .
```

### Adding as a C++ Dependency

Add silt as a submodule dependency to your repository:

```bash
git submodule add git@github.com:erosiv/silt.git ext/silt
git submodule update --init --recursive
```

Add the subdirectory to your `CMakeLists.txt` and link the library:

```CMakeLists.txt
add_subdirectory(${CMAKE_SOURCE_DIR}/ext/silt silt)
target_link_libraries(${TARGET_NAME} PUBLIC silt_lib)
```

Note that when you use `silt` as a dependency in a separate project's python bindings, it is interoperable with other libraries that use `silt`.

### Build Documentation

The documentation is build with sphinx:

```bash
sphinx-build doc build/html
```

## Why another tensor library?

`silt` was spun out of the tensor component of `soillib`, as more projects became dependent on it.

These projects have the same underlying goal: A polymorphic python interface for fast iteration and modular composition, with fully static C++ kernels for high performance. I found myself copy-pasting the same boilerplate over and over, so I decided to spin it out.

These projects also all share a similar build structure with similar goals: A CMake pipeline to build a C++/CUDA shared library, and python bindings with nanobind. With that in mind, this is designed to be included as a drop-in git submodule that *just works*.

I find that other projects are often unnecessarily large for monolithic kernel development, when all I really need is a small interface to convert data from various places on the python side, and provide a simple templated interface in C++.

Besides, sometimes it's just fun to roll your own. I would rather spend more time designing and building a small library like this than fighting build errors on somebody else's code.

## ToDo List

Find a way to further reduce the requirement for explicit template instantiations if possible.

Optional: Non-GLM Vector Types
  The ultimate goal is to FULLY eliminate the GLM types from the dependencies,
  because they are causing major portability issues withw windows for no benefit.

Optional: Orderings
  Introduce an ordering type that uses things like e.g. morton 
  order to turn a linear index into a non-linear index.

  Introduce an ordering type that allows for copy-free transposition

Optional: Sparsity
  Tensor Bags as more Complex Composed Maps
  Then, a map type can compose multiple of these together into weird structures including
  sparse structures, etc.