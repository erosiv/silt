# silt

`silt` is an isolated lightweight tensor library for easy inclusion in projects that use CUDA with Python bindings.

`silt` is primarily a light-weight wrapper for passing tensor data around between various libraries, and into kernels on the GPU for simulating physics.

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

### Python API

`silt` has two primary types: `shape` and `tensor`. Shape can have between 1 and 4 dimensions, while the tensor type supports multiple strict data-types and can live on the CPU or the GPU:

```python
s = silt.shape(512, 512)                    # 2D Tensor Shape
t = silt.tensor(s, silt.float32, silt.gpu)  # Strict-Typed GPU Tensor
silt.set(t, 0.0)                            # Set Data to Zeros
```

`silt` supports no-copy data wrapping to and from pytorch or numpy, as well as a simple data uploading downloading interface:

```python
t_numpy = silt.tensor.from_numpy(np.full((512, 512), 0.0, dtype=np.float32))                          # CPU Tensor
t_torch = silt.tensor.from_torch(torch.full((512, 512), 0.0, dtype=torch.flota32, device=torch.cuda)) # GPU Tensor

t_numpy = t_numpy.gpu() # Move data to GPU
t_torch = t_torch.cpu() # Move data to CPU

t_numpy = t_numpy.torch() # Convert to pytorch
t_torch = t_torch.numpy() # Convert to numpy
```

In essence, that is all `silt` does. The rest is providing the polymorphic conversion interface for strict-typed C++.

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