# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.20)
project(simd_compare LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_library(simd_kernels STATIC
    src/kernels_scalar.cpp
)
target_include_directories(simd_kernels PUBLIC include)
target_compile_options(simd_kernels PRIVATE -Wall -Wextra -Wpedantic)

# Detect target ISA family and add the matching SIMD translation unit.
# SSE: x86 / x86_64. We require SSSE3 for the RGB->gray shuffles.
# NEON: aarch64 has it unconditionally; armv7 needs -mfpu=neon.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|i[3-6]86)$")
    target_sources(simd_kernels PRIVATE src/kernels_sse.cpp)
    set_source_files_properties(src/kernels_sse.cpp PROPERTIES
        COMPILE_OPTIONS "-msse4.1;-mssse3;-msse2")
    target_compile_definitions(simd_kernels PUBLIC SIMD_COMPARE_HAVE_SSE=1)
    message(STATUS "simd_compare: building SSE kernels for ${CMAKE_SYSTEM_PROCESSOR}")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    target_sources(simd_kernels PRIVATE src/kernels_neon.cpp)
    target_compile_definitions(simd_kernels PUBLIC SIMD_COMPARE_HAVE_NEON=1)
    message(STATUS "simd_compare: building NEON kernels for ${CMAKE_SYSTEM_PROCESSOR}")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm.*)$")
    target_sources(simd_kernels PRIVATE src/kernels_neon.cpp)
    set_source_files_properties(src/kernels_neon.cpp PROPERTIES
        COMPILE_OPTIONS "-mfpu=neon")
    target_compile_definitions(simd_kernels PUBLIC SIMD_COMPARE_HAVE_NEON=1)
    message(STATUS "simd_compare: building NEON kernels for ${CMAKE_SYSTEM_PROCESSOR}")
else()
    message(STATUS "simd_compare: unknown ISA '${CMAKE_SYSTEM_PROCESSOR}', scalar only")
endif()

add_executable(simd_bench src/bench.cpp)
target_link_libraries(simd_bench PRIVATE simd_kernels)

enable_testing()
add_executable(simd_tests tests/test_kernels.cpp)
target_link_libraries(simd_tests PRIVATE simd_kernels)
add_test(NAME simd_kernels_correctness COMMAND simd_tests)
