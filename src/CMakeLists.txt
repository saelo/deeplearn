cmake_minimum_required(VERSION 3.8)
project(deeplearn LANGUAGES CXX)

set(CMAKE_BUILD_TYPE RelWithDebInfo)

find_package(OpenCL REQUIRED)

add_compile_options(-DCOPYGUARD -DDEBUG -pedantic -Wall)

include_directories(${OpenCL_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# TODO should probably use add_subdirectory here instead and create some static libs or similar for each directory ...
file(GLOB_RECURSE NN_Sources nn/*.cpp)
file(GLOB_RECURSE OCL_Sources ocl/*.cpp)
file(GLOB_RECURSE Util_Sources utils/*.cpp)

add_executable(deeplearn Main.cpp ${NN_Sources} ${OCL_Sources} ${Util_Sources})
target_link_libraries(deeplearn PRIVATE OpenCL::OpenCL)
set_target_properties(deeplearn PROPERTIES C_STANDARD 17
  C_STANDARD_REQUIRED ON
  C_EXTENSIONS OFF)
target_compile_definitions(deeplearn PRIVATE CL_TARGET_OPENCL_VERSION=120)
					       
# Build the test suite binary
add_executable(testsuite TestSuite.cpp ${NN_Sources} ${OCL_Sources} ${Util_Sources})
target_link_libraries(testsuite PRIVATE OpenCL::OpenCL)
set_target_properties(testsuite PROPERTIES C_STANDARD 17
  C_STANDARD_REQUIRED ON
  C_EXTENSIONS OFF)
target_compile_definitions(testsuite PRIVATE CL_TARGET_OPENCL_VERSION=120)

