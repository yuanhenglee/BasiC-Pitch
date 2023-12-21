#pragma once

#include "typedef.h"

int computeNFeaturesOut(int n_features_in, int kernel_size_feature, int stride);

Vectorf conv1d(Vectorf &x, Vectorf &filter_kernel, int stride);

Matrixf conv2d( const Matrixf &x, const Matrixf &filter_kernel, int stride );

Vectorf reflectionPadding(const Vectorf &x, int pad_length);

float sigmoid( float x );

// shape of input: (C, HW)
Matrixf im2col( const Matrixf& input,
    const int n_frames_in, const int n_features_in, const int n_frames_out, const int n_features_out,
    const int kernel_height, const int kernel_width, const int stride);

VecMatrixf col2im( const Matrixf& input, int n_frames_out, int n_features_out );