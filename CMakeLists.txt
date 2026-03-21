cmake_minimum_required(VERSION 3.25)
project(MetalFloat16Accelerator CXX)

# ── Platform targeting ─────────────────────────────────────────────────────────
set(CMAKE_SYSTEM_NAME Darwin)
set(CMAKE_OSX_ARCHITECTURES "arm64")
set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0")

# Generate compile_commands.json for clang-tidy / LSP
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── C++20 ─────────────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Include paths ─────────────────────────────────────────────────────────────
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ── Sanitizer option (pass -DSANITIZE=address|thread|undefined at configure) ──
option(SANITIZE "Enable sanitizer (address|thread|undefined|none)" "none")
if(NOT SANITIZE STREQUAL "none")
    add_compile_options(-fsanitize=${SANITIZE} -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=${SANITIZE})
    message(STATUS "Sanitizer: ${SANITIZE}")
endif()

# ── M2-specific compile flags (Release builds only) ───────────────────────────
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64" AND NOT SANITIZE STREQUAL "none")
    # Sanitizers need debug info; skip aggressive optimisations that obscure stacks
    add_compile_options(-O1 -mcpu=apple-m2)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    add_compile_options(
        -O3
        -ffast-math
        -mcpu=apple-m2
        -mtune=apple-m2
        -fno-math-errno
        -fno-trapping-math
    )
endif()

# ── Metal frameworks ──────────────────────────────────────────────────────────
find_library(METAL_LIBRARY     Metal)
find_library(FOUNDATION_LIBRARY Foundation)
find_library(QUARTZCORE_LIBRARY QuartzCore)

# ── Main shared library ───────────────────────────────────────────────────────
add_library(metal_float16_accelerator SHARED
    src/core/metal_device.cpp
    src/matrix/float16_matrix.cpp
    src/matrix/matrix_ops.cpp
    src/metal_float16_accelerator.cpp
)

target_link_libraries(metal_float16_accelerator
    PRIVATE
    ${METAL_LIBRARY}
    ${FOUNDATION_LIBRARY}
    ${QUARTZCORE_LIBRARY}
)

# ── Helper macro to add a binary linked against the library ───────────────────
macro(add_accelerator_exe TARGET_NAME SOURCE_FILE)
    add_executable(${TARGET_NAME} ${SOURCE_FILE})
    target_link_libraries(${TARGET_NAME} PRIVATE metal_float16_accelerator)
endmacro()

# ── Examples ──────────────────────────────────────────────────────────────────
add_accelerator_exe(basic_matmul_example      examples/basic_matmul.cpp)
add_accelerator_exe(simple_test               examples/simple_test.cpp)
add_accelerator_exe(speedup_demo              examples/speedup_demo.cpp)
add_accelerator_exe(alphazero_inference_sim   examples/alphazero_inference_sim.cpp)

# ── Benchmarks ────────────────────────────────────────────────────────────────
add_accelerator_exe(matrix_benchmarks  benchmarks/matrix_benchmarks.cpp)
add_accelerator_exe(latency_profile    benchmarks/latency_profile.cpp)

# ── Tests ─────────────────────────────────────────────────────────────────────
enable_testing()

add_accelerator_exe(correctness_tests tests/correctness_tests.cpp)

# Register with CTest so `ctest` / CI can drive them
add_test(NAME CorrectnessTests COMMAND correctness_tests)

# ── Install ───────────────────────────────────────────────────────────────────
install(
    TARGETS
        metal_float16_accelerator
        basic_matmul_example simple_test speedup_demo
        alphazero_inference_sim
        matrix_benchmarks latency_profile
        correctness_tests
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)
