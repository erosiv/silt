# silt

Silt is an isolated lightweight tensor library for easy inclusion in projects that use CUDA with Python bindings.

Note that this library doesn't implement complicated operations or features like autodifferentiation to provide a clean interface.

The goal is to be super lightweight with minimal compile times and easy inclusion into CMake projects.

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

### Install Python Bindings

Install `silt` using `pip`:

```bash
pip install .
```

No build-isolation progressive build (development):

```bash
pip install --no-build-isolation -ve .
```

Note that when you use `silt` as a dependency in a separate projects python bindings, it is interoperable with other libraries that use the dependency.

### ToDo List

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