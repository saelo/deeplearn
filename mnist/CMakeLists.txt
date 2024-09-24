cmake_minimum_required(VERSION 3.8)
project(clcldl LANGUAGES CXX)

set(CMAKE_BUILD_TYPE RelWithDebInfo)

find_package(OpenCL REQUIRED)

add_compile_options(-DDEBUG -pedantic -Wall)

include_directories(${OpenCL_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(../lib)

add_executable(deeplearn Main.cpp Mnist.cpp)
target_link_libraries(deeplearn PRIVATE clcldl)
set_target_properties(deeplearn PROPERTIES C_STANDARD 17
  C_STANDARD_REQUIRED ON
  C_EXTENSIONS OFF)
target_compile_definitions(deeplearn PRIVATE CL_TARGET_OPENCL_VERSION=120)
