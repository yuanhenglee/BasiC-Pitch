#include "utils.h"

#include <iostream>
#include <cmath>
#include <csignal>

void printMat(Matrixf &mat) {
    std::cout << mat << std::endl;
}

void printMat(Matrixcf &mat) {
    std::cout << mat << std::endl;
}

Vectorcf getHamming(int window_size) {
    Vectorcf window(window_size);
    for ( int i = 0 ; i < window_size ; i++ ) {
        window[i] = std::complex<float>(
            0.54f - 0.46f * cos(2.0f * M_PI * i / (window_size - 1)), 0.0f
        );
    }
    return window;
}

Vectorcf getHann(int window_size) {
    Vectorcf window(window_size);
    for ( int i = 0 ; i < window_size ; i++ ) {
        window[i] = std::complex<float>(
            0.5f * (1.0f - cos(2.0f * M_PI * i / (window_size - 1))), 0.0f
        );
    }
    return window;
}

inline int EDConut( float nyquist_hz, float filter_cutoff, int hop_length, int n_octaves ) {
    int count1 = std::max(0, 
        static_cast<int>(std::ceil(std::log2(0.85f * nyquist_hz / filter_cutoff))-2)
    );
    int count2 = std::max(0, 
        static_cast<int>(std::ceil(std::log2(hop_length)) - n_octaves + 1)
    );
    return std::min(count1, count2);
}

// utility function to update the CQParams, unused because early downsampling is not applied
void updateEDParams(CQParams &params) {
    
    float window_bandwidth = 1.5f;
    float filter_cutoff = params.fmax_t * (1.0f + 0.5f * window_bandwidth / params.quality_factor);
    float nyquist_hz = params.sample_rate / 2.0f;

    int downsample_count = EDConut(nyquist_hz, filter_cutoff, params.sample_per_frame, params.n_octaves);
    int downsample_factor = pow(2.0f, downsample_count);
    if ( downsample_count > 0 ) {
        params.sample_per_frame /= downsample_factor;
        params.sample_rate /= static_cast<float>(downsample_factor);
        params.downsample_factor = downsample_factor;
    }
}

Vectorf conv1d( Vectorf &x, Vectorf &filter_kernel, int stride ) {
    std::vector<float> result;
    for ( int i = 0 ; i + filter_kernel.size() <= x.size() ; i += stride ) {
        Vectorf temp = x.segment(i, filter_kernel.size());
        result.push_back(temp.dot(filter_kernel));
    }
    return Eigen::Map<Vectorf>(result.data(), result.size());
}

// The default filter_kernel is a low-pass filter if downsample_factor != 1
// x.shape = (n_samples)
// filter_kernel.shape = (1, kernel_length), default kernel_length = 256
// return_x.shape = (n_samples // 2)
Matrixf downsamplingByN(Vectorf &x, Vectorf &filter_kernel, float n) {
    int pad_length = (filter_kernel.cols() - 1) / 2;

    Vectorf padded_x = Vectorf::Zero(x.size() + 2 * pad_length);
    padded_x.segment(pad_length, x.size()) = x;
    return conv1d(padded_x, filter_kernel, static_cast<int>(n));
}

Vectorf reflectionPadding(const Vectorf &x, int pad_length) {
    Vectorf padded_x = Vectorf::Zero(x.size() + 2 * pad_length);
    padded_x.segment(pad_length, x.size()) = x;
    for ( int i = 0 ; i < pad_length ; i++ ) {
        padded_x[i] = x[pad_length - i];
        padded_x[padded_x.size() - 1 - i] = x[x.size() - 1 - pad_length + i];
    }
    return padded_x;
}

py::array_t<float> tensor2pyarray(Tensor3f &tensor) {
    std::vector<long int> shape = {tensor.dimension(0), tensor.dimension(1), tensor.dimension(2)};
    return py::array_t<float, py::array::c_style>(
        shape,
        tensor.data()
    );
}
