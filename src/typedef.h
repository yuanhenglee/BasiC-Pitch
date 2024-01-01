#pragma once

// #define EIGEN_DONT_PARALLELIZE
// #define USE_MKL
#ifdef USE_MKL
#define EIGEN_USE_MKL_ALL
#endif

#include "Eigen/Core"
#include "Eigen/Dense"
#include <vector>

typedef Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor> Vectorf;
typedef Eigen::Matrix<std::complex<float>, 1, Eigen::Dynamic, Eigen::RowMajor>
    Vectorcf;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    Matrixf;
typedef Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    Matrixcf;
typedef std::vector<Matrixf, Eigen::aligned_allocator<Matrixf>> VecMatrixf;
typedef std::vector<VecMatrixf, Eigen::aligned_allocator<VecMatrixf>> VecVecMatrixf;
