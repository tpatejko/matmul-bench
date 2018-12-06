cmake_minimum_required(VERSION 2.8)
project(test-shuffle)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include(CheckCXXCompilerFlag)

if(UNIX)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
add_definitions(-std=c++11)
elseif(COMPILER_SUPPORTS_CXX0X)
add_definitions(-std=c++0x)
endif()
endif()

set(MKLDNNROOT "$ENV{MKLDNNROOT}")
set(MKLROOT "$ENV{MKLROOT}")

find_path(MKLDNN_INCLUDE_DIR mkldnn.hpp PATHS ${MKLDNNROOT} PATH_SUFFIXES include)

find_library(MKLDNN_LIBRARY
        NAMES mkldnn
        PATHS ${MKLDNNROOT}/lib)

message("MKLDNN_LIBRARY: ${MKLDNN_LIBRARY}")

if(MKLDNNROOT)
  message(STATUS "Found MKLDNN (include: ${MKLDNN_INCLUDE_DIR}, lib: ${MKLDNN_LIBRARY}")
  include_directories(${MKLDNN_INCLUDE_DIR})
else()
  message(FATAL "Intel MKLDNN not located properly")
endif()

find_path(MKL_INCLUDE_DIR mkl_cblas.h PATHS ${MKLROOT} PATH_SUFFIXES include)
find_library(MKL_LIBRARY
             NAMES mklml_intel
             PATHS ${MKLROOT}/lib)
if(MKLROOT)
  message(STATUS "Found MKL (include: ${MKL_INCLUDE_DIR}")
  include_directories(${MKL_INCLUDE_DIR})
else()
  message(FATAL "MKL not found")
endif()

add_executable(test-matmul bench.cpp)

add_definitions(-Wall)

target_link_libraries(test-matmul ${MKLDNN_LIBRARY} ${MKL_LIBRARY})
