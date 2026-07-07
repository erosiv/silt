#
# silt C++ Library Target
#

set(TARGET_NAME silt_lib)
add_library(${TARGET_NAME} SHARED)
add_library(silt::silt ALIAS ${TARGET_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES 
  LINKER_LANGUAGE CUDA
  CUDA_RUNTIME_LIBRARY Shared
)

target_compile_definitions(${TARGET_NAME} PRIVATE SILT_SHARED_BUILD)
target_include_directories(${TARGET_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/silt>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_sources(${TARGET_NAME} PRIVATE
  silt/core/memory.cu
  silt/op/common_unary.cu
  silt/op/common_binary.cu
  silt/op/uncommon.cu
  silt/op/indexed.cu
)

if(NOT TARGET glm::glm)
  add_subdirectory(../ext/glm glm)
endif()
target_link_libraries(${TARGET_NAME} PRIVATE glm::glm)