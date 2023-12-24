#pragma once

#include "typedef.h"

int computeNFeaturesOut(int n_features_in, int kernel_size_feature, int stride);

Vectorf conv1d(Vectorf &x, Vectorf &filter_kernel, int stride);

Matrixf conv2d( const Matrixf &x, const Matrixf &filter_kernel, int stride );

Vectorf reflectionPadding(const Vectorf &x, int pad_length);

// shape of input: (H, W)
Matrixf im2col( const VecMatrixf& input, int n_frames_out, int n_features_out, int kernel_height, int kernel_width, int stride);

VecMatrixf col2im( const Matrixf& input, int n_frames_out, int n_features_out );