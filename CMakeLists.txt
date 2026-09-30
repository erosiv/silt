cmake_minimum_required(VERSION 3.21.0)

#
# Load Version
#
# project() requires a strictly numeric major.minor.patch, so VERSION must not
# carry a PEP 440 pre-release suffix such as .dev0 or rc1.

file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/VERSION" SILT_VERSION LIMIT_COUNT 1)

#
# Project Configuration
#

set(PROJECT_NAME silt)
project(${PROJECT_NAME} VERSION ${SILT_VERSION} LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
#set(CMAKE_CUDA_ARCHITECTURES 80 86 87 89 90 100 120)
set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --extended-lambda")

if(MSVC)
  # cccl (bundled from CUDA 13.2 on) requires MSVC's conforming preprocessor;
  # the traditional one is only a warning under 13.1 but a hard error
  # (C1189) under 13.2. Switch globally rather than suppress the warning.
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=\"/Zc:preprocessor\"")
endif()

#
# Project Components
#

add_subdirectory(source)    # Build Library
if(PROJECT_IS_TOP_LEVEL)
add_subdirectory(python)  # Build Bindings
endif()

# Testing off by default. Opt in explicitly with -DSILT_BUILD_TESTS=ON
option(SILT_BUILD_TESTS "Build silt's own C++ unit tests (see test/cpp)" OFF)
if(SILT_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test/cpp)
endif()