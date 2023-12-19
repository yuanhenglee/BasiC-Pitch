#pragma once

#include "Eigen/Core"
#include "Eigen/Sparse"
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
