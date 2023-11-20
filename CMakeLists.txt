cmake_minimum_required(VERSION 3.12)
project(BasiCPP_Pitch VERSION 1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Global settings
set(LDFLAGS "-O3 -fopenmp -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${LDFLAGS}")

# fftw
set(FFTW_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/src/ext/fftw-3.3.10/build )
link_libraries(${FFTW_ROOT}/lib/libfftw3f.so)
include_directories(${FFTW_ROOT}/include)

# Eigen
# set(EIGEN_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/src/ext/eigen-3.3.7)
set(EIGEN_ROOT /usr/include/eigen3)
include_directories(${EIGEN_ROOT})

# Find Python and pybind11 package
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

# Set sources for the library
set(SOURCES
    ./src/main.cpp
    ./src/CQT.cpp
    ./src/utils.cpp
)

# Set so library output path
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)

# Create the shared library
pybind11_add_module(BasiCPP_Pitch MODULE ${SOURCES})

# Create so library
target_compile_definitions(BasiCPP_Pitch PRIVATE VERSION_INFO=${PROJECT_VERSION})


