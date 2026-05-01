# Tests CMakeLists.txt

cmake_minimum_required(VERSION 3.16)

# Find Google Test
find_package(GTest CONFIG REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Test executables
set(TEST_SOURCES
    test_attention.cpp
    test_cache.cpp
    test_inference.cpp
    test_compression.cpp
)

foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME} 
        PRIVATE 
            optimization_core
            GTest::gtest
            GTest::gtest_main
    )
    
    # Add test
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    
    # Set properties
    set_target_properties(${TEST_NAME} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )
endforeach()

# Custom target for all tests
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS ${TEST_SOURCES}
)












