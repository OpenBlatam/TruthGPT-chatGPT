cmake_minimum_required(VERSION 3.0)
project(YourProjectName)

# Find Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Add your source files
add_executable(YourExecutableName your_source_files.cpp)

# Link Eigen
target_link_libraries(YourExecutableName Eigen3::Eigen)
