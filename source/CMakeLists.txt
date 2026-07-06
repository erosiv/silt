#
# silt C++ Library Target
#

set(TARGET_NAME silt_lib)
add_library(${TARGET_NAME} SHARED)
set_target_properties(${TARGET_NAME} PROPERTIES 
  LINKER_LANGUAGE CUDA
)
target_compile_definitions(${TARGET_NAME} PRIVATE -DSHARED_BUILD)
target_include_directories(${TARGET_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/silt>
  .
)

target_sources(${TARGET_NAME} PUBLIC
  silt/op/common_unary.cu
  silt/op/common_binary.cu
  silt/op/uncommon.cu
  silt/op/indexed.cu
)

add_subdirectory(../ext/glm glm)
target_link_libraries(${TARGET_NAME} PRIVATE glm::glm)