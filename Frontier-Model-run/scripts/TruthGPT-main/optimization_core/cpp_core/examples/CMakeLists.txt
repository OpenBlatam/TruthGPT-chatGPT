# Examples CMakeLists.txt

cmake_minimum_required(VERSION 3.16)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Example executables
set(EXAMPLE_SOURCES
    example_attention.cpp
    example_cache.cpp
    example_inference.cpp
    example_compression.cpp
)

foreach(EXAMPLE_SOURCE ${EXAMPLE_SOURCES})
    get_filename_component(EXAMPLE_NAME ${EXAMPLE_SOURCE} NAME_WE)
    add_executable(${EXAMPLE_NAME} ${EXAMPLE_SOURCE})
    target_link_libraries(${EXAMPLE_NAME} 
        PRIVATE 
            optimization_core
    )
    
    # Set properties
    set_target_properties(${EXAMPLE_NAME} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )
endforeach()

# Custom target for all examples
add_custom_target(examples
    DEPENDS ${EXAMPLE_SOURCES}
)












