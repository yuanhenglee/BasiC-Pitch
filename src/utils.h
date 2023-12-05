#pragma once

#include "typedef.h"
#include "CQT.h"

#include <iostream>

void printMat(Matrixf &mat);

void printMat(Matrixcf &mat);

Vectorcf getHamming(int window_size);

Vectorcf getHann(int window_size);

void updateEDParams(CQParams &params);

Vectorf conv1d(Vectorf &x, Vectorf &filter_kernel, int stride);

Matrixf downsamplingByN(Vectorf &x, Vectorf &filter_kernel, float n);

Vectorf reflectionPadding(const Vectorf &x, int pad_length);

py::array_t<float> tensor2pyarray(Tensor3f &tensor);

Tensor3f pyarray2tensor(py::array_t<float> &pyarray);

template <typename... Dims>
Tensor2f matrix2Tensor(Matrixf &matrix, Dims... dims) {
    return Eigen::TensorMap<Tensor2f>(matrix.data(), {dims...});
}

int computeNFeaturesOut(int n_features_in, int kernel_size_feature, int stride);