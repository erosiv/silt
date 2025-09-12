# template-cmake-cuda-py

Template repository for simple python bindings through CMake with CUDA.

## Structure and Dependencies

```yaml
ext: nanobind dependency as git submodule
lib: C++ library dependencies (header / source)
python: Python Binding Code Specification
```

Note that the C++ dependencies can be specified compiled individually as targets,
individually of the python bindings.

- Install Python3 (w. development libraries) w. pip
- Install CMake
- [Windows] Install Visual Studio 17 2022 Build Tools
- [Unix] Install GCC, Gnu Make

Initialize nanobind as git submodule.

```bash
git submodule update --init --recursive
```

## Direct Installation

Compile and install the python module directly using `pip`:

```bash
pip install .
```

### Building from Scratch

#### Windows

Build with CMake:

```bash
cmake -S . -B build  -G "Visual Studio 17 2022"
cmake --build build
```