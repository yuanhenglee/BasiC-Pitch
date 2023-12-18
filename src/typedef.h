#pragma once

#include "Eigen/Core"
#include "constant.h"
#include <vector>

typedef Eigen::Matrix<float, 1, Eigen::Dynamic> Vectorf;
typedef Eigen::Matrix<std::complex<float>, 1, Eigen::Dynamic>
    Vectorcf;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>
    Matrixf;
typedef Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic>
    Matrixcf;
typedef std::vector<Matrixf> VecMatrixf;
typedef std::vector<VecMatrixf> VecVecMatrixf;

// static size for the CQT kernel
typedef Eigen::Matrix<std::complex<float>, CQT_KERNEL_HEIGHT, CQT_KERNEL_WIDTH>
    CQTKernelMat;