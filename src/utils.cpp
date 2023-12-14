#include "utils.h"
#include "constant.h"

#include <iostream>
#include <cmath>
#include <csignal>

void printMat(Matrixf &mat) {
    std::cout << mat << std::endl;
}

void printMat(Matrixcf &mat) {
    std::cout << mat << std::endl;
}

void printVecMatrixf(VecMatrixf &tensor) {
    for ( int i = 0 ; i < tensor.size() ; i++ ) {
        std::cout << "i = " << i << std::endl;
        printMat(tensor[i]);
    }
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

// NOTE: use VALID padding as default
Vectorf conv1d( Vectorf &x, Vectorf &filter_kernel, int stride ) {
    std::vector<float> result;
    for ( int i = 0 ; i + filter_kernel.size() <= x.size() ; i += stride ) {
        Vectorf temp = x.segment(i, filter_kernel.size());
        result.push_back(temp.dot(filter_kernel));
    }
    return Eigen::Map<Vectorf>(result.data(), result.size());
}

inline int padLength(int input_length, int filter_length, int stride, int output_length) {
    return (output_length-1) * stride + filter_length - input_length;
}

// NOTE: use SAME padding as default
// x.shape = (n_samples, n_features_in)
// NOTE: Since we are dealing with audio signals, the number of samples remains the same
Matrixf conv2d( const Matrixf &x, const Matrixf &filter_kernel, int stride ) {
// void conv2d( const Matrixf &x, const Matrixf &filter_kernel, int stride, Matrixf &result ) {
    int n_samples_out = x.rows();
    int n_features_out = computeNFeaturesOut(x.cols(), filter_kernel.cols(), stride);
    int pad_height = padLength(x.rows(), filter_kernel.rows(), 1, n_samples_out);
    int pad_width = padLength(x.cols(), filter_kernel.cols(), stride, n_features_out);
    Matrixf result(n_samples_out, n_features_out);
    Matrixf padded_x = Matrixf::Zero(x.rows() + pad_height, x.cols() + pad_width);
    padded_x.block(pad_height / 2, pad_width / 2, x.rows(), x.cols()) = x;
    
    Matrixf temp(filter_kernel.rows(), filter_kernel.cols());
    for ( int i = 0 ; i < n_samples_out ; i++ ) {
        for ( int j = 0 ; j < n_features_out ; j++ ) {
            temp = padded_x.block(i , j * stride, filter_kernel.rows(), filter_kernel.cols());
            result(i, j) = (temp.cwiseProduct(filter_kernel)).sum();
        }
    }    

    return result;
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

int computeNFeaturesOut(int n_features_in, int kernel_size_feature, int stride) {
    // padding == "same"
    float f = static_cast<float>(n_features_in) / static_cast<float>(stride);
    return std::ceil(f);
}

std::vector<Vectorf> getWindowedAudio( const Vectorf &x ) {

    int audio_length = x.size() + OVERLAP_LENGTH / 2;
    int n_windows = std::ceil(static_cast<float>(audio_length) / static_cast<float>(WINDOW_HOP_SIZE));
    int padded_length = (n_windows - 1) * WINDOW_HOP_SIZE + AUDIO_N_SAMPLES;
    // add padding to the audio signal
    Vectorf padded_x = Vectorf::Zero(padded_length);
    padded_x.segment(OVERLAP_LENGTH / 2, x.size()) = x;
    
    std::vector<Vectorf> result;
    for ( int i = 0 ; i + AUDIO_N_SAMPLES <= padded_length ; i += WINDOW_HOP_SIZE ) {
        Vectorf temp = padded_x.segment(i, AUDIO_N_SAMPLES);
        result.emplace_back(temp);
    }

    return result;
}

Matrixf concatMatrices(const VecMatrixf &matrices, int audio_length ) {
    int n_frames_original = std::floor(audio_length * (ANNOTATIONS_FPS * 1.0f /SAMPLE_RATE) );
    int mat_height = matrices[0].rows() - N_OVERLAP_FRAMES;
    int mat_width = matrices[0].cols();
    Matrixf result = Matrixf::Zero( matrices.size() * mat_height, mat_width );
    int start = 0;
    for ( const Matrixf& m : matrices ) {
        // remove half of the overlap frames from the beginning and the end
        result.block(start, 0, 
            mat_height, mat_width
            ) = 
            m.block(N_OVERLAP_FRAMES/2, 0, 
            mat_height, mat_width
        );
        start += mat_height;
    }
    return result.block(0, 0, n_frames_original, mat_width);
}