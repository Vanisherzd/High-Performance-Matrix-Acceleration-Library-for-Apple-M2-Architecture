cmake_minimum_required(VERSION 3.25)
project(MetalFloat16Accelerator CXX)

# M2 Apple Silicon targeting
set(CMAKE_SYSTEM_NAME Darwin)
set(CMAKE_OSX_ARCHITECTURES "arm64")
set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# M2-specific optimizations
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    add_compile_options(
            -O3
            -ffast-math
            -mcpu=apple-m2
            -mtune=apple-m2
            -fno-math-errno
            -fno-trapping-math
        )
endif()

# Main library target
add_library(metal_float16_accelerator SHARED
    src/core/metal_device.cpp
    src/matrix/float16_matrix.cpp
    src/matrix/matrix_ops.cpp
    src/metal_float16_accelerator.cpp
)

# Find required frameworks
find_library(METAL_LIBRARY Metal)
find_library(FOUNDATION_LIBRARY Foundation)
find_library(QUARTZCORE_LIBRARY QuartzCore)

# Link Metal frameworks
target_link_libraries(metal_float16_accelerator
    PRIVATE
    ${METAL_LIBRARY}
    ${FOUNDATION_LIBRARY}
    ${QUARTZCORE_LIBRARY}
)

# Example executable
add_executable(basic_matmul_example
    examples/basic_matmul.cpp
)

target_link_libraries(basic_matmul_example
    PRIVATE
    metal_float16_accelerator
)

# Simple test executable
add_executable(simple_test
    examples/simple_test.cpp
)

target_link_libraries(simple_test
    PRIVATE
    metal_float16_accelerator
)

# Benchmark executable
add_executable(matrix_benchmarks
    benchmarks/matrix_benchmarks.cpp
)

target_link_libraries(matrix_benchmarks
    PRIVATE
    metal_float16_accelerator
)

# Install targets
install(TARGETS metal_float16_accelerator basic_matmul_example simple_test matrix_benchmarks
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib)