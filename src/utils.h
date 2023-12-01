#ifndef UTILS_H
#define UTILS_H

#include "Eigen/Core"
#include "Eigen/Sparse"
#include "CQT.h"
#include <iostream>

typedef Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor> Vectorf;
typedef Eigen::Matrix<std::complex<float>, 1, Eigen::Dynamic, Eigen::RowMajor>
    Vectorcf;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    Matrixf;
typedef Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    Matrixcf;

void printMat(Matrixf &mat);

void printMat(Matrixcf &mat);

Vectorcf getHamming(int window_size);

Vectorcf getHann(int window_size);

void updateEDParams(CQParams &params);

Vectorf defaultLowPassFilter();

Vectorf conv1d(Vectorf &x, Vectorf &filter_kernel, int stride);

Matrixf downsamplingByN(Vectorf &x, Vectorf &filter_kernel, float n);

void loadDefaultKernel(Matrixcf &kernel);

Vectorf reflectionPadding(const Vectorf &x, int pad_length);

#endif // UTILS_H