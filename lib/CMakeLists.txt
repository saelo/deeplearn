cmake_minimum_required(VERSION 3.8)
project(clcldl LANGUAGES CXX)

find_package(OpenCL REQUIRED)

include_directories(${OpenCL_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB_RECURSE NN_Sources nn/*.cpp)
file(GLOB_RECURSE OCL_Sources ocl/*.cpp)
file(GLOB_RECURSE Util_Sources utils/*.cpp)

add_library(clcldl
  STATIC
  ${NN_Sources}
  ${OCL_Sources}
  ${Util_Sources}
  )
target_compile_definitions(clcldl PRIVATE CL_TARGET_OPENCL_VERSION=120)
target_link_libraries(clcldl PRIVATE OpenCL::OpenCL)
