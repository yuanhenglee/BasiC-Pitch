#pragma once

#include "typedef.h"
#include "CQT.h"
#include <vector>

void printMat(Matrixf &mat);

void printMat(Matrixcf &mat);

// void printPyarray(py::array_t<float> &pyarray);

void printVecMatrixf(VecMatrixf &tensor);

Vectorcf getHamming(int window_size);

Vectorcf getHann(int window_size);

void updateEDParams(CQParams &params);

Vectorf conv1d(Vectorf &x, Vectorf &filter_kernel, int stride);

Matrixf conv2d( const Matrixf &x, const Matrixf &filter_kernel, int stride );

Matrixf downsamplingByN(Vectorf &x, Vectorf &filter_kernel, float n);

Vectorf reflectionPadding(const Vectorf &x, int pad_length);

// py::array_t<float> mat3D2pyarray(VecMatrixf &tensor);

// VecMatrixf pyarray2mat3D(py::array_t<float> &pyarray);

int computeNFeaturesOut(int n_features_in, int kernel_size_feature, int stride);

std::vector<Vectorf> getWindowedAudio(const Vectorf &x);

Matrixf concatMatrices(const VecMatrixf &matrices, int audio_length);