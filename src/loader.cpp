#pragma once

#include "loader.h"
#include "json.hpp"
#include "cnpy.h"

void loadDefaultKernel(Matrixcf &kernel) {
    // load the precomputed kernel
    cnpy::NpyArray arr = cnpy::npy_load("model/kernel.npy");
    const size_t& n_bins = arr.shape[0];
    const size_t& kernel_length = arr.shape[1];
    std::complex<float>* data = const_cast<std::complex<float>*>(arr.data<std::complex<float>>());
    kernel = Eigen::Map<Matrixcf>(data, n_bins, kernel_length);
}

void loadDefaultLowPassFilter( Vectorf &filter_kernel) {
    // load the precomputed filter
    cnpy::NpyArray arr = cnpy::npy_load("model/lowpass_filter.npy");
    const size_t& kernel_length = arr.shape[0];
    float* data = const_cast<float*>(arr.data<float>());
    filter_kernel = Eigen::Map<Vectorf>(data, kernel_length);
}