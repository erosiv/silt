#
# Compiled and Linked Library
#

# Dependency Files

add_library(my_cuda_lib SHARED)
set_target_properties(my_cuda_lib PROPERTIES 
  LINKER_LANGUAGE CUDA
)

target_compile_definitions(my_cuda_lib PRIVATE -DSHARED_BUILD)
target_include_directories(my_cuda_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/my_cuda_lib>
  .
)

# target_include_directories(my_cuda_lib PRIVATE
#   ${CUDA_INCLUDE_DIRS}
# )

target_sources(my_cuda_lib PUBLIC
  my_cuda_lib/my_source.cu
  # ... other source files ...
)