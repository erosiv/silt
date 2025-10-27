Using silt
==========

Fundamentally, the goal of silt is to enable the building of modular tensor accelerated libraries without heavy infrastructure or modifications of your build system. Therefore, silt has been designed to be *trivially includable* in your CMake + CUDA project.

Building a CUDA Library
-----------------------

1. Import git submodule:

silt is most easily imported into your project as a git submodule:

.. code::
  bash

  git submodule add git@github.com:erosiv/silt.git ext/silt
  git submodule update --init --recursive

2. Add to CMake:

silt can then be built by adding it as a subdirectory to your ``CMakeLists.txt``.

.. code::
  CMake

  add_subdirectory(${CMAKE_SOURCE_DIR}/ext/silt silt)

This will build and expose two targets, ``silt_lib`` which is a shared C++ library, and ``silt`` which is a python module.

3. Link to your C++ Library:

Finally, you can make the headers available and link silt to your own C++ library in CMake:

.. code::
  CMake

  target_link_libraries(${TARGET_NAME} PUBLIC silt_lib)

Full CMake Example
^^^^^^^^^^^^^^^^^^

.. code::
  CMake

  # my_lib CMakeLists.txt Example

  set(TARGET_NAME my_lib)
  add_library(${TARGET_NAME} STATIC)
  set_target_properties(${TARGET_NAME} PROPERTIES 
    LINKER_LANGUAGE CUDA
  )

  target_include_directories(${TARGET_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/source>
    .
  )

  target_sources(${TARGET_NAME} PUBLIC
    # ... .cpp source files ...
  )

  # ... other dependencies ...

  add_subdirectory(${CMAKE_SOURCE_DIR}/ext/silt silt)
  target_link_libraries(${TARGET_NAME} PUBLIC silt_lib)

Building Python Bindings
------------------------

When building python bindings that expose functions which accept silt types, you can either link silt explicitly in CMake, or you can link your C++ library.

The silt repository itself generates the python bindings for silt, you can inspect ``CMakeLists.txt`` to see an example linking structure.

Note that when you use `silt` as a dependency in a separate project's python bindings, it is interoperable with other libraries that use `silt`. This is possible because of ``dllexport`` directives on the primary exposed type: the tensor.