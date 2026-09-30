#
# silt C++ unit tests.
#
# Covers ownership/lifetime behaviour of tensor_t<T>/view_t<T> that is
# not reachable from Python -- only the polymorphic `tensor`, `view`,
# `shape` and `slice` wrappers are bound (see doc/extending.rst and
# python/source/*.cpp), so none of tensor_t<T>'s refcounting, its
# self-assignment behaviour, or the move-vs-clone distinction between
# tensor_t<T> and `tensor` can be exercised from there.
#
# This target is CPU-only by design: these are ownership/lifetime
# properties, independent of which device the data lives on, so no CUDA
# device is required to build or run it. It is fetched, built and run
# only when SILT_BUILD_TESTS is on (the default when silt is the
# top-level project; off for a submodule consumer), so it never affects
# a downstream project that just wants silt_lib.
#

include(FetchContent)
FetchContent_Declare(
  doctest
  GIT_REPOSITORY https://github.com/doctest/doctest.git
  GIT_TAG        v2.4.11
)

# doctest 2.4.11's own CMakeLists.txt declares cmake_minimum_required(VERSION
# 3.0...), and CMake 4.x removed support for running project files that only
# claim compatibility below 3.5. CMAKE_POLICY_VERSION_MINIMUM (CMake >= 3.31)
# tells CMake to treat a fetched project's policies as at least this version,
# without needing -DCMAKE_POLICY_VERSION_MINIMUM on the command line. Scoped to
# just this FetchContent call via save/restore so it doesn't change policy
# behaviour for silt's own CMakeLists.txt files.
set(_silt_saved_policy_min "${CMAKE_POLICY_VERSION_MINIMUM}")
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
FetchContent_MakeAvailable(doctest)
set(CMAKE_POLICY_VERSION_MINIMUM "${_silt_saved_policy_min}")
unset(_silt_saved_policy_min)

add_executable(silt_test_cpp
  test_tensor.cpp
  test_gather_scatter.cpp
  test_indexed.cpp
  test_index_set.cpp
  test_sort.cpp
  test_histogram.cpp
)

# Deliberately does NOT define HAS_CUDA before including silt headers,
# matching how python/source/*.cpp compiles them today (see
# doc/extending.rst, "Host and device functions"). This test target is a
# plain host executable; it links against the already-built silt_lib
# rather than compiling any silt source itself.
target_link_libraries(silt_test_cpp PRIVATE silt::silt doctest::doctest)

# silt_lib is SHARED, and its .dll lands in a different output directory
# than silt_test_cpp.exe under a multi-config generator (Visual Studio
# puts each target's Debug/Release output under its own subdirectory).
# Windows has no rpath -- it only searches the executable's own directory
# and PATH -- so without this, the test exits immediately at launch with
# STATUS_DLL_NOT_FOUND (0xc0000135) whenever ctest runs it standalone.
# Copy every DLL silt_test_cpp actually depends on next to it after each
# build; TARGET_RUNTIME_DLLS resolves transitively and is a no-op
# (empty list) on platforms that don't need it.
if(MSVC)
add_custom_command(TARGET silt_test_cpp POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "$<TARGET_RUNTIME_DLLS:silt_test_cpp>"
    "$<TARGET_FILE_DIR:silt_test_cpp>"
  COMMAND_EXPAND_LISTS
)
endif()

include(CTest)
include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)
doctest_discover_tests(silt_test_cpp)
