# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.15)
if(POLICY CMP0148)
  cmake_policy(SET CMP0148 OLD)
endif()

project ("TheSimulator" VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type" FORCE)

# C++ standard
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CXX_STANDARD 17)

# Find required packages at top level
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Include sub-projects.
add_subdirectory ("TheSimulator")