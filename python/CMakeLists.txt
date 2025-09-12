#
# Python Binding Generation
#

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

#
# Load Dependencies
#

find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
add_subdirectory(${CMAKE_SOURCE_DIR}/ext/nanobind nanobind)

#
# Define the Module and Bindings
#

set(MODULE_NAME my_bindings)

nanobind_add_module(${MODULE_NAME}
  source/module.cpp 
  source/component.cpp
  NB_SHARED
)

target_compile_definitions(${MODULE_NAME} PRIVATE
  MODULE_NAME=${MODULE_NAME})

# Internal Target Dependencies
target_link_libraries(${MODULE_NAME} PRIVATE my_cuda_lib)  # Internal Dependency

# Install Directive for Scikit
install(TARGETS ${MODULE_NAME} LIBRARY DESTINATION ${MODULE_NAME})
install(TARGETS nanobind RUNTIME DESTINATION ${SKBUILD_PLATLIB_DIR}/my_bindings)
install(TARGETS my_cuda_lib RUNTIME DESTINATION ${SKBUILD_PLATLIB_DIR}/my_bindings)