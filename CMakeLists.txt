cmake_minimum_required(VERSION 3.16)
project(mm_test_library CXX)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

# Compute Library
add_library (mm STATIC
    src/mm/blas1.cpp
    src/mm/blas2.cpp
    src/mm/gemm.cpp)

target_include_directories(mm PUBLIC include/mm)

set(MM_LOG_LEVEL 0 CACHE STRING "mm log level: 0=off 1=info(GEMM) 2=debug(BLAS-2) 3=trace(BLAS-1)")
target_compile_definitions(mm PRIVATE MM_LOG_LEVEL=${MM_LOG_LEVEL})

# Testing
include(${CMAKE_SOURCE_DIR}/cmake/dependencies.cmake)

enable_testing()

add_executable (test_mm_correctness test/test_mm_correctness.cpp)
target_link_libraries (test_mm_correctness PRIVATE mm GTest::gtest)
include(GoogleTest)
gtest_discover_tests(test_mm_correctness)

add_executable(mm_bench bench/mm_bench.cpp)
target_link_libraries (mm_bench PRIVATE mm benchmark::benchmark)

# Examples: one runnable sample per variant
foreach(variant ijp ipj jip jpi pij pji)
    add_executable(sample_mm_${variant} examples/mm/sample_mm_${variant}.cpp)
    target_link_libraries(sample_mm_${variant} PRIVATE mm)
endforeach()
