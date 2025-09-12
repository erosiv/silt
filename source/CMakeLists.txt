#
# soillib C++ Library Target
#

set(TARGET_NAME silt_lib)
add_library(${TARGET_NAME} SHARED)
set_target_properties(${TARGET_NAME} PROPERTIES 
  LINKER_LANGUAGE CUDA
)
target_compile_definitions(${TARGET_NAME} PRIVATE -DSHARED_BUILD)
target_include_directories(${TARGET_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/silt>
  .
)

target_sources(${TARGET_NAME} PUBLIC
  silt/op/common.cu
)

find_package(glm CONFIG REQUIRED)
target_link_libraries(${TARGET_NAME} PRIVATE glm::glm)