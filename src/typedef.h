#pragma once

#include "Eigen/Core"
#include "Eigen/Sparse"
#include "unsupported/Eigen/FFT"
#include "unsupported/Eigen/CXX11/Tensor"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

typedef Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor> Vectorf;
typedef Eigen::Matrix<std::complex<float>, 1, Eigen::Dynamic, Eigen::RowMajor>
    Vectorcf;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    Matrixf;
typedef Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    Matrixcf;
typedef Eigen::Tensor<float, 3, Eigen::RowMajor> Tensor3f;
typedef Eigen::Tensor<float, 2, Eigen::RowMajor> Tensor2f;

namespace py = pybind11;
