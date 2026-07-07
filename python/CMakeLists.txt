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

target_link_libraries(${MODULE_NAME} PRIVATE glm::glm)
target_link_libraries(${MODULE_NAME} PRIVATE silt::silt)

#
# Installation
#

# Install Directive for Scikit
install(TARGETS ${MODULE_NAME} LIBRARY DESTINATION ${MODULE_NAME})

# Install Runtime Dependencies into Wheel File
# install(TARGETS nanobind RUNTIME DESTINATION ${SKBUILD_PLATLIB_DIR}/${MODULE_NAME})
install(TARGETS silt_lib RUNTIME DESTINATION ${SKBUILD_PLATLIB_DIR}/${MODULE_NAME})