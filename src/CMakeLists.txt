cmake_minimum_required(VERSION 3.8)
project(deeplearn)

include(CheckCXXCompilerFlag)
set(EXTRA_COMPILE_FLAGS "-Wall")
CHECK_CXX_COMPILER_FLAG(-std=c++11 HAS_CXX_11)
if(HAS_CXX_11)
    set(EXTRA_COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS} -std=c++11")
else(HAS_CXX_11)
    message(Error "C++11 support is required.")
endif()
set (CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${EXTRA_COMPILE_FLAGS}")

find_package(OpenCL REQUIRED)

include_directories(${OpenCL_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# TODO should probably use add_subdirectory here instead and create some static libs or similar for each directory ...
file(GLOB_RECURSE NN_Sources nn/*.cpp)
file(GLOB_RECURSE OCL_Sources ocl/*.cpp)
file(GLOB_RECURSE Util_Sources utils/*.cpp)

add_executable(deeplearn Main.cpp ${NN_Sources} ${OCL_Sources} ${Util_Sources})
target_link_libraries(deeplearn ${OpenCL_LIBRARIES})

# Build the test suite binary
add_executable(testsuite TestSuite.cpp ${NN_Sources} ${OCL_Sources} ${Util_Sources})
target_link_libraries(testsuite ${OpenCL_LIBRARIES})

# Add a #define to indicate Debug builds
set(CMAKE_C_FLAGS_DEBUG "-g -DDEBUG -DCOPYGUARD -fsanitize=address")
set(CMAKE_CXX_FLAGS_DEBUG "-g -DDEBUG -DCOPYGUARD -fsanitize=address")
set(CMAKE_BUILD_TYPE Release)
