# Benchmarks CMakeLists.txt

cmake_minimum_required(VERSION 3.16)

# Find Google Benchmark
find_package(benchmark CONFIG REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Benchmark executable
add_executable(benchmark_all benchmark_all.cpp)
target_link_libraries(benchmark_all 
    PRIVATE 
        optimization_core
        benchmark::benchmark
        benchmark::benchmark_main
)

# Attention-specific benchmark
add_executable(benchmark_attention benchmark_attention.cpp)
target_link_libraries(benchmark_attention 
    PRIVATE 
        optimization_core
        benchmark::benchmark
        benchmark::benchmark_main
)

# Set properties
set_target_properties(benchmark_all benchmark_attention PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Custom target
add_custom_target(run_benchmarks
    COMMAND benchmark_all --benchmark_format=console
    DEPENDS benchmark_all
)












