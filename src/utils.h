#pragma once

#include "typedef.h"
#include "CQT.h"
#include <vector>

void printMat(Matrixf &mat);

void printMat(Matrixcf &mat);

void printVecMatrixf(VecMatrixf &tensor);

Vectorcf getHamming(int window_size);

Vectorcf getHann(int window_size);

void updateEDParams(CQParams &params);

Matrixf downsamplingByN(Vectorf &x, Vectorf &filter_kernel, float n);

std::vector<Vectorf> getWindowedAudio(const Vectorf &x);

Matrixf concatMatrices(const VecMatrixf &matrices, const int audio_length, const int n_frames_in);