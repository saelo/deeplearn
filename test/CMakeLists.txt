cmake_minimum_required(VERSION 3.8)
project(clcldl LANGUAGES CXX)

include_directories(${OpenCL_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(../lib)

# Build the test suite binary
add_executable(testsuite TestSuite.cpp)
target_link_libraries(testsuite PRIVATE clcldl)
set_target_properties(testsuite PROPERTIES C_STANDARD 17
  C_STANDARD_REQUIRED ON
  C_EXTENSIONS OFF)
target_compile_definitions(testsuite PRIVATE CL_TARGET_OPENCL_VERSION=120)

add_test(TestItAll testsuite)
