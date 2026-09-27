#
# silt C++ Library Target
#

set(TARGET_NAME silt_lib)
add_library(${TARGET_NAME} SHARED)
add_library(silt::silt ALIAS ${TARGET_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES 
  LINKER_LANGUAGE CUDA
  CUDA_RUNTIME_LIBRARY Static
)

target_compile_definitions(${TARGET_NAME} PRIVATE SILT_SHARED_BUILD)
target_compile_definitions(${TARGET_NAME} PUBLIC HAS_CUDA)

# Version macros, for downstream #if guarding. PUBLIC so that anything linking
# silt_lib sees them. SILT_VERSION_NUM is the comparable form, e.g.
#   #if SILT_VERSION_NUM >= 10200
math(EXPR SILT_VERSION_NUM
  "${silt_VERSION_MAJOR} * 10000 + ${silt_VERSION_MINOR} * 100 + ${silt_VERSION_PATCH}")
target_compile_definitions(${TARGET_NAME} PUBLIC
  SILT_VERSION_MAJOR=${silt_VERSION_MAJOR}
  SILT_VERSION_MINOR=${silt_VERSION_MINOR}
  SILT_VERSION_PATCH=${silt_VERSION_PATCH}
  SILT_VERSION_NUM=${SILT_VERSION_NUM}
  SILT_VERSION_STRING="${silt_VERSION}"
)

target_include_directories(${TARGET_NAME} PUBLIC
#  $<BUILD_INTERFACE:${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/silt>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_sources(${TARGET_NAME} PRIVATE
  silt/core/memory.cu
  silt/op/common_unary.cu
  silt/op/common_binary.cu
  silt/op/reduce.cu
  silt/op/indexed.cu
  silt/op/indexed_reduce.cu
  silt/op/rng.cu
)

if(NOT TARGET glm::glm)
  add_subdirectory(../ext/glm glm)
endif()
target_link_libraries(${TARGET_NAME} PUBLIC glm::glm)

find_package(CUDAToolkit)
target_link_libraries(${TARGET_NAME} PUBLIC CUDA::curand)