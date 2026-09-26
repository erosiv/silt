#
# Python Binding Generation
#

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

#
# Define the Module and Bindings
#

find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
if(NOT TARGET nanobind::nanobind_static AND NOT TARGET nanobind)
  add_subdirectory(${CMAKE_SOURCE_DIR}/ext/nanobind nanobind)
endif()

set(MODULE_NAME silt)
nanobind_add_module(${MODULE_NAME}
  NB_SHARED
  NB_DOMAIN "erosiv"
  source/_module.cpp
  source/shape.cpp
  source/slice.cpp
  source/tensor.cpp
  source/view.cpp
  source/op.cpp
  source/util.cpp
  source/indexed.cpp
)

target_compile_definitions(${MODULE_NAME} PRIVATE
  MODULE_NAME=${MODULE_NAME})

#
# Dependencies
#

#
# Type Stubs (.pyi)
#
#  Generates python/silt/${MODULE_NAME}.pyi next to __init__.py by actually
#  importing the just-built extension (PYTHON_PATH points at wherever CMake
#  put it) and introspecting it -- this is why it needs the module to have
#  built and be loadable, not just parseable source. Runs as part of the
#  normal build (nanobind_add_stub adds it to the ALL target), and writes
#  straight into the source tree so both an editable install (pip install
#  -ve .) and a full wheel build pick it up -- the latter via
#  wheel.packages picking up python/silt wholesale in pyproject.toml. The
#  file itself is generated, not committed (see .gitignore); py.typed next
#  to it is the actual, committed PEP 561 marker that tells type checkers
#  to look for it at all.
#
nanobind_add_stub(
  ${MODULE_NAME}_stub
  MODULE ${MODULE_NAME}
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/silt/${MODULE_NAME}.pyi
  PYTHON_PATH $<TARGET_FILE_DIR:${MODULE_NAME}>
  DEPENDS ${MODULE_NAME}
)

# glm::glm transitively linked through silt
target_link_libraries(${MODULE_NAME} PRIVATE silt::silt)

#
# Installation
#

# Install Directive for scikit-build module
install(TARGETS ${MODULE_NAME} LIBRARY DESTINATION ${MODULE_NAME})

install(TARGETS silt_lib
  RUNTIME DESTINATION ${SKBUILD_PLATLIB_DIR}/${MODULE_NAME}
  LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR}/${MODULE_NAME}
)

install(TARGETS nanobind-erosiv
  RUNTIME DESTINATION ${SKBUILD_PLATLIB_DIR}/${MODULE_NAME}
  LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR}/${MODULE_NAME}
)