cmake_minimum_required(VERSION 3.12)
project(BasiCPP_Pitch VERSION 1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Global settings
set(LDFLAGS "-O3 -fopenmp -fno-math-errno -march=native -DNDEBUG -w")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${LDFLAGS}")

# cnpy
set( CNPY_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/src/ext/cnpy )
include_directories(${CNPY_ROOT})

# nlohmann json
set( JSON_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/src/ext/nlohmann-json )
include_directories(${JSON_ROOT})

# fftw
set(FFTW_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/src/ext/fftw-3.3.10/build )
link_libraries(${FFTW_ROOT}/lib/libfftw3f.so)
include_directories(${FFTW_ROOT}/include)

# Eigen
set(EIGEN_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/src/ext/eigen-3.3.7)
# set(EIGEN_ROOT /usr/include/eigen3)
include_directories(${EIGEN_ROOT})

# Find Python and pybind11 package
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

find_package(ZLIB REQUIRED)
include_directories(${ZLIB_INCLUDE_DIRS})

# Set sources for the library
# set(SOURCES
#     ./src/main.cpp
#     ./src/CQT.cpp
#     ./src/utils.cpp
#     ./src/loader.cpp
#     ./src/ext/cnpy/cnpy.cpp
# )
file(GLOB SOURCES
    ./src/*.cpp
    ./src/ext/cnpy/cnpy.cpp
)

# Set headers for the library
# set(HEADERS
#     ./src/typedef.h
#     ./src/CQT.h
#     ./src/utils.h
#     ./src/loader.h
#     ./src/ext/cnpy/cnpy.h
#     ./src/ext/nlohmann-json/json.hpp
# )
file(GLOB HEADERS
    ./src/*.h
    ./src/ext/cnpy/cnpy.h
    ./src/ext/nlohmann-json/json.hpp
)

# Set so library output path
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)

# Create the shared library
pybind11_add_module(BasiCPP_Pitch MODULE ${SOURCES} ${HEADERS})

# Create so library
target_compile_definitions(BasiCPP_Pitch PRIVATE VERSION_INFO=${PROJECT_VERSION})


