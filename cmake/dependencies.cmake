# Third-party dependencies fetched via FetchContent.
# Included from the top-level CMakeLists.txt via include().

include(FetchContent)

# GoogleTest
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.17.0                                   # latest release
  GIT_SHALLOW    TRUE
  SOURCE_DIR     ${CMAKE_SOURCE_DIR}/third_party/googletest
)

# GoogleBenchmark
set(BENCHMARK_ENABLE_TESTING      OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS  OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG        v1.9.5                                    # latest release
  GIT_SHALLOW    TRUE
  SOURCE_DIR     ${CMAKE_SOURCE_DIR}/third_party/benchmark
)

FetchContent_MakeAvailable(googletest benchmark)
