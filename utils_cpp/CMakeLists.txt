cmake_minimum_required(VERSION 2.8.3)
project(utils_cpp_py)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

# Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

add_subdirectory(pybind11)
pybind11_add_module(utils_cpp_py
        utilsCppPy.cpp)

target_link_libraries(utils_cpp_py
        PRIVATE)

