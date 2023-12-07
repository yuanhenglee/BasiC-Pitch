#pragma once

#include "Eigen/Core"
#include "Eigen/Sparse"
#include "unsupported/Eigen/FFT"
// #include "unsupported/Eigen/CXX11/Tensor"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <vector>

typedef Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor> Vectorf;
typedef Eigen::Matrix<std::complex<float>, 1, Eigen::Dynamic, Eigen::RowMajor>
    Vectorcf;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    Matrixf;
typedef Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    Matrixcf;
typedef std::vector<Matrixf> VecMatrixf;
typedef std::vector<VecMatrixf> VecVecMatrixf;

namespace py = pybind11;
