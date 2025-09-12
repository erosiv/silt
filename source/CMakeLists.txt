#
# soillib C++ Library Target
#

add_library(silt SHARED)
set_target_properties(silt PROPERTIES 
  LINKER_LANGUAGE CUDA
)
target_compile_definitions(silt PRIVATE -DSHARED_BUILD)
target_include_directories(silt PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/silt>
  .
)

target_sources(silt PUBLIC
  soillib/op/common.cu
)

find_package(glm CONFIG REQUIRED)
target_link_libraries(silt PRIVATE glm::glm)