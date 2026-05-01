# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  optimization_core C++ Extensions                                             ║
# ║  High-performance C++ backend for TruthGPT optimization                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

cmake_minimum_required(VERSION 3.21)

# ═══════════════════════════════════════════════════════════════════════════════
# PROJECT DEFINITION
# ═══════════════════════════════════════════════════════════════════════════════

project(optimization_core_cpp 
    VERSION 1.1.0
    DESCRIPTION "High-performance C++ extensions for TruthGPT"
    HOMEPAGE_URL "https://github.com/truthgpt/optimization_core"
    LANGUAGES CXX
)

# Provide useful message
message(STATUS "")
message(STATUS "╔══════════════════════════════════════════════════════════════════╗")
message(STATUS "║  optimization_core C++ Extensions v${PROJECT_VERSION}                    ║")
message(STATUS "╚══════════════════════════════════════════════════════════════════╝")

# ═══════════════════════════════════════════════════════════════════════════════
# CMAKE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Set default build type if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS 
        "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

# Modern CMake settings
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Use ccache if available
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    message(STATUS "Using ccache: ${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

# Include CMake modules
include(CheckCXXCompilerFlag)
include(CheckLanguage)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(FetchContent)

# ═══════════════════════════════════════════════════════════════════════════════
# BUILD OPTIONS
# ═══════════════════════════════════════════════════════════════════════════════

option(BUILD_PYTHON_BINDINGS "Build Python bindings with PyBind11" ON)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)
option(BUILD_EXAMPLES "Build examples" OFF)

# Feature flags
option(USE_EIGEN "Use Eigen for linear algebra" ON)
option(USE_ONEDNN "Use oneDNN for deep learning primitives" ON)
option(USE_TBB "Use Intel TBB for parallelization" ON)
option(USE_MIMALLOC "Use mimalloc for memory allocation" ON)
option(USE_CUTLASS "Use NVIDIA CUTLASS for GPU kernels" ON)
option(USE_GGML "Use ggml for tensor operations" OFF)
option(USE_COMPRESSION "Enable LZ4/ZSTD compression" ON)
option(USE_XSIMD "Use xsimd for portable SIMD" ON)
option(USE_FMT "Use fmt for formatting" ON)
option(USE_SPDLOG "Use spdlog for logging" ON)

# Debug options
option(ENABLE_SANITIZERS "Enable address/undefined sanitizers (Debug only)" OFF)
option(ENABLE_COVERAGE "Enable code coverage" OFF)
option(ENABLE_LTO "Enable Link-Time Optimization (Release only)" ON)
option(ENABLE_NATIVE_ARCH "Enable native architecture optimizations" ON)

# ═══════════════════════════════════════════════════════════════════════════════
# COMPILER FLAGS & SIMD DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

# Compiler-specific flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-g -O0)
    else()
        add_compile_options(-O3)
        if(ENABLE_NATIVE_ARCH)
            add_compile_options(-march=native -mtune=native)
        endif()
    endif()
    
    # Enable LTO for Release builds
    if(ENABLE_LTO AND CMAKE_BUILD_TYPE MATCHES "Release|RelWithDebInfo")
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    endif()
    
elseif(MSVC)
    add_compile_options(/W4 /permissive- /Zc:__cplusplus)
    
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(/Od /Zi)
    else()
        add_compile_options(/O2 /GL)
        if(ENABLE_NATIVE_ARCH)
            add_compile_options(/arch:AVX2)
        endif()
    endif()
endif()

# SIMD detection
set(SIMD_FLAGS "")
set(SIMD_DEFINITIONS "")

check_cxx_compiler_flag("-mavx512f" HAVE_AVX512)
check_cxx_compiler_flag("-mavx2" HAVE_AVX2)
check_cxx_compiler_flag("-msse4.2" HAVE_SSE42)
check_cxx_compiler_flag("-mfma" HAVE_FMA)

if(HAVE_AVX512)
    list(APPEND SIMD_DEFINITIONS "HAVE_AVX512=1")
    message(STATUS "SIMD: AVX-512 detected")
elseif(HAVE_AVX2)
    list(APPEND SIMD_DEFINITIONS "HAVE_AVX2=1")
    message(STATUS "SIMD: AVX2 detected")
elseif(HAVE_SSE42)
    list(APPEND SIMD_DEFINITIONS "HAVE_SSE42=1")
    message(STATUS "SIMD: SSE4.2 detected")
endif()

if(HAVE_FMA)
    list(APPEND SIMD_DEFINITIONS "HAVE_FMA=1")
endif()

# ARM NEON detection
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    check_cxx_compiler_flag("-march=armv8-a+simd" HAVE_NEON)
    if(HAVE_NEON)
        list(APPEND SIMD_DEFINITIONS "HAVE_NEON=1")
        message(STATUS "SIMD: ARM NEON detected")
    endif()
endif()

# Sanitizers (Debug only)
if(ENABLE_SANITIZERS AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
        add_link_options(-fsanitize=address,undefined)
        message(STATUS "Sanitizers: Address + UBSan enabled")
    endif()
endif()

# Code coverage
if(ENABLE_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(--coverage -fprofile-arcs -ftest-coverage)
        add_link_options(--coverage)
        message(STATUS "Coverage: Enabled")
    endif()
endif()

# ═══════════════════════════════════════════════════════════════════════════════
# CUDA SUPPORT
# ═══════════════════════════════════════════════════════════════════════════════

check_language(CUDA)
if(CMAKE_CUDA_COMPILER AND USE_CUTLASS)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    
    # Detect CUDA architecture
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # Default to common architectures
        set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")
    endif()
    
    add_compile_definitions(HAVE_CUDA)
    message(STATUS "CUDA: ${CMAKE_CUDA_COMPILER_VERSION} (archs: ${CMAKE_CUDA_ARCHITECTURES})")
else()
    message(STATUS "CUDA: Not available")
endif()

# ═══════════════════════════════════════════════════════════════════════════════
# FIND DEPENDENCIES
# ═══════════════════════════════════════════════════════════════════════════════

# Threads (always required)
find_package(Threads REQUIRED)

# Eigen (Header-only linear algebra)
if(USE_EIGEN)
    find_package(Eigen3 3.4 CONFIG)
    if(Eigen3_FOUND)
        add_compile_definitions(HAVE_EIGEN)
        message(STATUS "Eigen: ${Eigen3_VERSION}")
    else()
        message(STATUS "Eigen: Not found (will use fallback)")
    endif()
endif()

# Intel oneDNN (Deep learning primitives)
if(USE_ONEDNN)
    find_package(dnnl CONFIG)
    if(dnnl_FOUND)
        add_compile_definitions(HAVE_ONEDNN)
        message(STATUS "oneDNN: Found")
    else()
        message(STATUS "oneDNN: Not found")
    endif()
endif()

# Intel TBB (Parallelization)
if(USE_TBB)
    find_package(TBB CONFIG)
    if(TBB_FOUND)
        add_compile_definitions(HAVE_TBB)
        message(STATUS "TBB: Found")
    else()
        # Fallback to OpenMP
        find_package(OpenMP)
        if(OpenMP_CXX_FOUND)
            add_compile_definitions(HAVE_OPENMP)
            message(STATUS "TBB: Not found, using OpenMP")
        else()
            message(STATUS "TBB/OpenMP: Not found")
        endif()
    endif()
endif()

# mimalloc (High-performance allocator)
if(USE_MIMALLOC)
    find_package(mimalloc CONFIG)
    if(mimalloc_FOUND)
        add_compile_definitions(HAVE_MIMALLOC)
        message(STATUS "mimalloc: Found")
    else()
        message(STATUS "mimalloc: Not found")
    endif()
endif()

# xsimd (Portable SIMD)
if(USE_XSIMD)
    find_package(xsimd CONFIG)
    if(xsimd_FOUND)
        add_compile_definitions(HAVE_XSIMD)
        message(STATUS "xsimd: Found")
    else()
        message(STATUS "xsimd: Not found")
    endif()
endif()

# LZ4 (Fast compression)
if(USE_COMPRESSION)
    find_package(lz4 CONFIG)
    if(lz4_FOUND OR TARGET lz4::lz4)
        add_compile_definitions(HAVE_LZ4)
        message(STATUS "LZ4: Found")
    else()
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(LZ4 IMPORTED_TARGET liblz4)
            if(LZ4_FOUND)
                add_compile_definitions(HAVE_LZ4)
                message(STATUS "LZ4: Found (pkg-config)")
            endif()
        endif()
    endif()
    
    # Zstandard (Better compression ratio)
    find_package(zstd CONFIG)
    if(zstd_FOUND OR TARGET zstd::libzstd_static)
        add_compile_definitions(HAVE_ZSTD)
        message(STATUS "ZSTD: Found")
    else()
        message(STATUS "ZSTD: Not found")
    endif()
endif()

# fmt (Formatting library)
if(USE_FMT)
    find_package(fmt CONFIG)
    if(fmt_FOUND)
        add_compile_definitions(HAVE_FMT)
        message(STATUS "fmt: ${fmt_VERSION}")
    else()
        message(STATUS "fmt: Not found")
    endif()
endif()

# spdlog (Fast logging)
if(USE_SPDLOG)
    find_package(spdlog CONFIG)
    if(spdlog_FOUND)
        add_compile_definitions(HAVE_SPDLOG)
        message(STATUS "spdlog: ${spdlog_VERSION}")
    else()
        message(STATUS "spdlog: Not found")
    endif()
endif()

# NVIDIA CUTLASS (GPU kernels)
if(USE_CUTLASS AND CMAKE_CUDA_COMPILER)
    find_path(CUTLASS_INCLUDE_DIR cutlass/cutlass.h
        HINTS ${CUTLASS_ROOT} ENV CUTLASS_ROOT
        PATH_SUFFIXES include
    )
    if(CUTLASS_INCLUDE_DIR)
        add_compile_definitions(HAVE_CUTLASS)
        message(STATUS "CUTLASS: ${CUTLASS_INCLUDE_DIR}")
    else()
        message(STATUS "CUTLASS: Not found")
    endif()
endif()

# PyBind11 (Python bindings)
if(BUILD_PYTHON_BINDINGS)
    find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)
    message(STATUS "Python: ${Python3_VERSION}")
    message(STATUS "pybind11: ${pybind11_VERSION}")
endif()

# ═══════════════════════════════════════════════════════════════════════════════
# INCLUDE DIRECTORIES
# ═══════════════════════════════════════════════════════════════════════════════

set(INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include  # Generated headers
)

if(CUTLASS_INCLUDE_DIR)
    list(APPEND INCLUDE_DIRS ${CUTLASS_INCLUDE_DIR})
endif()

# ═══════════════════════════════════════════════════════════════════════════════
# LIBRARY SOURCES
# ═══════════════════════════════════════════════════════════════════════════════

set(CORE_SOURCES
    src/memory/allocator.cpp
    src/attention/attention_kernels.cpp
    src/optim/optimizers.cpp
    src/inference/engine.cpp
)

set(CORE_HEADERS
    include/attention/flash_attention.hpp
    include/memory/kv_cache.hpp
)

# CUDA sources (if available)
if(CMAKE_CUDA_COMPILER)
    list(APPEND CORE_SOURCES
        src/attention/flash_attention_cuda.cu
        src/memory/gpu_allocator.cu
    )
endif()

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LIBRARY TARGET
# ═══════════════════════════════════════════════════════════════════════════════

add_library(optimization_core_cpp ${CORE_SOURCES})
add_library(optimization_core::core ALIAS optimization_core_cpp)
add_library(optimization_core ALIAS optimization_core_cpp)  # Convenience alias

target_include_directories(optimization_core_cpp
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}/include
)

# Apply SIMD definitions
target_compile_definitions(optimization_core_cpp PRIVATE ${SIMD_DEFINITIONS})

# Link dependencies
target_link_libraries(optimization_core_cpp PUBLIC Threads::Threads)

if(Eigen3_FOUND)
    target_link_libraries(optimization_core_cpp PUBLIC Eigen3::Eigen)
endif()

if(dnnl_FOUND)
    target_link_libraries(optimization_core_cpp PUBLIC DNNL::dnnl)
endif()

if(TBB_FOUND)
    target_link_libraries(optimization_core_cpp PUBLIC TBB::tbb)
elseif(OpenMP_CXX_FOUND)
    target_link_libraries(optimization_core_cpp PUBLIC OpenMP::OpenMP_CXX)
endif()

if(mimalloc_FOUND)
    target_link_libraries(optimization_core_cpp PUBLIC mimalloc)
endif()

if(xsimd_FOUND)
    target_link_libraries(optimization_core_cpp PUBLIC xsimd)
endif()

if(TARGET lz4::lz4)
    target_link_libraries(optimization_core_cpp PUBLIC lz4::lz4)
elseif(TARGET PkgConfig::LZ4)
    target_link_libraries(optimization_core_cpp PUBLIC PkgConfig::LZ4)
endif()

if(TARGET zstd::libzstd_static)
    target_link_libraries(optimization_core_cpp PUBLIC zstd::libzstd_static)
elseif(TARGET zstd::libzstd_shared)
    target_link_libraries(optimization_core_cpp PUBLIC zstd::libzstd_shared)
endif()

if(fmt_FOUND)
    target_link_libraries(optimization_core_cpp PUBLIC fmt::fmt)
endif()

if(spdlog_FOUND)
    target_link_libraries(optimization_core_cpp PUBLIC spdlog::spdlog)
endif()

# Set library properties
set_target_properties(optimization_core_cpp PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# ═══════════════════════════════════════════════════════════════════════════════
# PYTHON BINDINGS
# ═══════════════════════════════════════════════════════════════════════════════

if(BUILD_PYTHON_BINDINGS)
    set(PYBIND_SOURCES
        python/bindings.cpp
    )
    
    pybind11_add_module(_cpp_core MODULE ${PYBIND_SOURCES})
    
    target_link_libraries(_cpp_core PRIVATE optimization_core_cpp)
    target_compile_definitions(_cpp_core PRIVATE 
        VERSION_INFO="${PROJECT_VERSION}"
    )
    
    # Enable if we want to strip symbols
    if(CMAKE_BUILD_TYPE MATCHES "Release|RelWithDebInfo")
        set_target_properties(_cpp_core PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION ON
        )
    endif()
    
    # Install Python module
    install(TARGETS _cpp_core
        LIBRARY DESTINATION ${Python3_SITELIB}/optimization_core
        RUNTIME DESTINATION ${Python3_SITELIB}/optimization_core
    )
endif()

# ═══════════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════════

if(BUILD_TESTS)
    enable_testing()
    find_package(GTest CONFIG)
    
    if(GTest_FOUND)
        # Test executable helper function
        function(add_cpp_test test_name test_source)
            add_executable(${test_name} ${test_source})
            target_link_libraries(${test_name} PRIVATE
                optimization_core_cpp
                GTest::gtest_main
                GTest::gmock
            )
            target_include_directories(${test_name} PRIVATE ${INCLUDE_DIRS})
            add_test(NAME ${test_name} COMMAND ${test_name})
            
            # Enable test discovery
            include(GoogleTest)
            gtest_discover_tests(${test_name})
        endfunction()
        
        # Add all test files
        file(GLOB TEST_SOURCES tests/test_*.cpp)
        foreach(TEST_SOURCE ${TEST_SOURCES})
            get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
            add_cpp_test(${TEST_NAME} ${TEST_SOURCE})
        endforeach()
        
        message(STATUS "Tests: Enabled")
    else()
        message(STATUS "Tests: GTest not found, skipping")
    endif()
endif()

# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARKS
# ═══════════════════════════════════════════════════════════════════════════════

if(BUILD_BENCHMARKS)
    find_package(benchmark CONFIG)
    
    if(benchmark_FOUND)
        function(add_cpp_benchmark bench_name bench_source)
            add_executable(${bench_name} ${bench_source})
            target_link_libraries(${bench_name} PRIVATE
                optimization_core_cpp
                benchmark::benchmark
                benchmark::benchmark_main
            )
            target_include_directories(${bench_name} PRIVATE ${INCLUDE_DIRS})
        endfunction()
        
        # Add all benchmark files
        file(GLOB BENCH_SOURCES benchmarks/benchmark_*.cpp)
        foreach(BENCH_SOURCE ${BENCH_SOURCES})
            get_filename_component(BENCH_NAME ${BENCH_SOURCE} NAME_WE)
            add_cpp_benchmark(${BENCH_NAME} ${BENCH_SOURCE})
        endforeach()
        
        message(STATUS "Benchmarks: Enabled")
    else()
        message(STATUS "Benchmarks: Google Benchmark not found, skipping")
    endif()
endif()

# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLES
# ═══════════════════════════════════════════════════════════════════════════════

if(BUILD_EXAMPLES)
    function(add_cpp_example example_name example_source)
        add_executable(${example_name} ${example_source})
        target_link_libraries(${example_name} PRIVATE optimization_core_cpp)
        target_include_directories(${example_name} PRIVATE ${INCLUDE_DIRS})
    endfunction()
    
    # Add all example files
    file(GLOB EXAMPLE_SOURCES examples/example_*.cpp)
    foreach(EXAMPLE_SOURCE ${EXAMPLE_SOURCES})
        get_filename_component(EXAMPLE_NAME ${EXAMPLE_SOURCE} NAME_WE)
        add_cpp_example(${EXAMPLE_NAME} ${EXAMPLE_SOURCE})
    endforeach()
    
    message(STATUS "Examples: Enabled")
endif()

# ═══════════════════════════════════════════════════════════════════════════════
# INSTALLATION
# ═══════════════════════════════════════════════════════════════════════════════

install(TARGETS optimization_core_cpp
    EXPORT optimization_core_cpp-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING 
        PATTERN "*.hpp"
        PATTERN "*.h"
)

install(EXPORT optimization_core_cpp-targets
    FILE optimization_core_cpp-targets.cmake
    NAMESPACE optimization_core::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/optimization_core_cpp
)

# ═══════════════════════════════════════════════════════════════════════════════
# PACKAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/optimization_core_cpp-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/optimization_core_cpp-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/optimization_core_cpp
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/optimization_core_cpp-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/optimization_core_cpp-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/optimization_core_cpp-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/optimization_core_cpp
)

# ═══════════════════════════════════════════════════════════════════════════════
# UNINSTALL TARGET
# ═══════════════════════════════════════════════════════════════════════════════

if(NOT TARGET uninstall)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
        IMMEDIATE @ONLY
    )
    
    add_custom_target(uninstall
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake
    )
endif()

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

message(STATUS "")
message(STATUS "╔══════════════════════════════════════════════════════════════════╗")
message(STATUS "║  Configuration Summary                                           ║")
message(STATUS "╠══════════════════════════════════════════════════════════════════╣")
message(STATUS "║  Build Type:          ${CMAKE_BUILD_TYPE}")
message(STATUS "║  C++ Standard:        C++${CMAKE_CXX_STANDARD}")
message(STATUS "║  Compiler:            ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "╠══════════════════════════════════════════════════════════════════╣")
message(STATUS "║  Libraries:                                                      ║")
message(STATUS "║    Eigen:             ${Eigen3_FOUND}")
message(STATUS "║    oneDNN:            ${dnnl_FOUND}")
message(STATUS "║    TBB:               ${TBB_FOUND}")
message(STATUS "║    OpenMP:            ${OpenMP_CXX_FOUND}")
message(STATUS "║    mimalloc:          ${mimalloc_FOUND}")
message(STATUS "║    xsimd:             ${xsimd_FOUND}")
message(STATUS "║    fmt:               ${fmt_FOUND}")
message(STATUS "║    spdlog:            ${spdlog_FOUND}")
message(STATUS "╠══════════════════════════════════════════════════════════════════╣")
message(STATUS "║  GPU:                                                            ║")
message(STATUS "║    CUDA:              ${CMAKE_CUDA_COMPILER}")
message(STATUS "║    CUTLASS:           ${CUTLASS_INCLUDE_DIR}")
message(STATUS "╠══════════════════════════════════════════════════════════════════╣")
message(STATUS "║  Targets:                                                        ║")
message(STATUS "║    Python bindings:   ${BUILD_PYTHON_BINDINGS}")
message(STATUS "║    Tests:             ${BUILD_TESTS}")
message(STATUS "║    Benchmarks:        ${BUILD_BENCHMARKS}")
message(STATUS "╠══════════════════════════════════════════════════════════════════╣")
message(STATUS "║  Features:                                                       ║")
message(STATUS "║    Sanitizers:        ${ENABLE_SANITIZERS}")
message(STATUS "║    Coverage:          ${ENABLE_COVERAGE}")
message(STATUS "║    LTO:               ${ENABLE_LTO}")
message(STATUS "║    Native arch:       ${ENABLE_NATIVE_ARCH}")
message(STATUS "╚══════════════════════════════════════════════════════════════════╝")
message(STATUS "")
