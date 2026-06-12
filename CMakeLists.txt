cmake_minimum_required(VERSION 3.14)
project(pln_ir_initialization)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# This assumes pybind11 is installed, e.g., via pip install pybind11
# For cmake to find it, you might need to run:
# cmake -DCMAKE_PREFIX_PATH=$(python3 -m pybind11 --includes) ..
find_package(pybind11 REQUIRED)

pybind11_add_module(initialization_cpp 
    src/cpp/initialization.cpp
    src/cpp/bindings.cpp
)
# Add NMSLIB
set(WITHOUT_TESTS ON)
add_subdirectory(external/nmslib/similarity_search)

target_include_directories(initialization_cpp PRIVATE 
    src/cpp/include 
    external/nmslib/similarity_search/include
    external/falconn/src/include
    external/falconn/external/eigen
    external/falconn/external/simple-serializer
)
target_link_libraries(initialization_cpp PRIVATE flann_cpp lz4 NonMetricSpaceLib)

# Set output directory so the module is found automatically by the Python scripts
set_target_properties(initialization_cpp PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/src/python"
)
