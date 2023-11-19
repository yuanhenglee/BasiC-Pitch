cmake_minimum_required(VERSION 3.12)
project(BasiCPP_Pitch VERSION 1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11 package
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

# Set sources for the library
set(SOURCES
    ./src/main.cpp
    ./src/CQT.cpp
)

# Set so library output path
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)

# Create the shared library
pybind11_add_module(BasiCPP_Pitch MODULE ${SOURCES})

# Create so library
target_compile_definitions(BasiCPP_Pitch PRIVATE VERSION_INFO=${PROJECT_VERSION})


