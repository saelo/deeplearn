cmake_minimum_required(VERSION 3.8)
project(clcldl LANGUAGES CXX)

set(CMAKE_BUILD_TYPE RelWithDebInfo)

find_package(OpenCL REQUIRED)

add_compile_options(-DCOPYGUARD -DDEBUG -pedantic -Wall)

include_directories(${OpenCL_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

enable_testing()

add_subdirectory(lib)
add_subdirectory(mnist)
add_subdirectory(test)
