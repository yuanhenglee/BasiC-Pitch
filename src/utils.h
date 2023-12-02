#ifndef UTILS_H
#define UTILS_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "Eigen/Core"
#include "Eigen/Sparse"
#include "unsupported/Eigen/CXX11/Tensor"
#include "CQT.h"
#include <iostream>

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

py::array_t<float> tensor2pyarray(Tensor3f &tensor);

template <typename... Dims>
Tensor2f matrix2Tensor(Matrixf &matrix, Dims... dims) {
    return Eigen::TensorMap<Tensor2f>(matrix.data(), {dims...});
}

#endif // UTILS_H