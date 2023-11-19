#include "CQT.h"
// #include "utils.h"
#include "Eigen/Core"
#include "Eigen/Sparse"
#include "unsupported/Eigen/FFT"
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <vector>

static std::vector<float> getHammingWindow(int window_size) {
    std::vector<float> window(window_size);
    for ( int i = 0 ; i < window_size ; i++ ) {
        window[i] = 0.54f - 0.46f * cos(2.0f * M_PI * i / (window_size - 1));
    }
    return window;
}

static std::vector<float> getArangeWindow(int window_size) {
    int start = -(window_size - 1) / 2, end = (window_size - 1) / 2;
    std::vector<float> window;
    window.reserve(window_size);
    for ( int i = start ; i <= end ; i++ ) {
        window.push_back(i);
    }
    return window;
}

CQParams::CQParams(float sample_rate, int bins_per_octave, float freq_min, float freq_max, int n_bins)
    : sample_rate(sample_rate), bins_per_octave(bins_per_octave), freq_min(freq_min), freq_max(freq_max), n_bins(n_bins) {  
    n_freq = round(bins_per_octave * log2(freq_max / freq_min));
    quality_factor = 1.0f / (pow(2.0f, 1.0f / bins_per_octave) - 1.0f);
    fft_window_size = static_cast<int>(pow(2.0f, 
        ceil(
            log2(quality_factor * sample_rate / freq_min) + 1e-6
        ))
    );
}   

CQ::CQ(CQParams params) : params(params) {
    _kernel = Eigen::SparseMatrix<std::complex<float>>(params.n_freq, params.fft_window_size);
    computeKernel();
}

CQ::~CQ() {
    std::cout<<"CQ destructor"<<std::endl;
}

void CQ::computeKernel() {
    // std::cout<<"CQ::runKernel()"<<std::endl;
    Matrixcf kernel = Matrixcf::Zero(params.n_freq, params.fft_window_size);
    for ( int i = 0 ; i < params.n_freq ; i++ ) {

        float freq = params.freq_min * pow(2.0f, i / params.bins_per_octave);
        int window_length = 2 * round(
            params.quality_factor * params.sample_rate / freq / 2.0f
        ) + 1;
        
        Vectorcf temporal_kernel = Vectorcf::Zero(window_length);

        std::vector<float> hamming_window = getHammingWindow(window_length);
        std::vector<float> arange_window = getArangeWindow(window_length);

        for ( int j = 0 ; j < window_length ; j++ ) {
            std::complex<float> temp = std::complex<float>(
                0.0f, 2.0f * M_PI * params.quality_factor * arange_window[j] / window_length
            );
            temporal_kernel[j] = std::exp(temp) * 
                std::complex<float>(hamming_window[j] / window_length, 0.0f);
        }

        int pad_length = (params.fft_window_size - window_length + 1) / 2;

        kernel.block(i, pad_length, 1, window_length) = temporal_kernel;
    }

    Eigen::FFT<float> fft;
    for ( int i = 0 ; i < params.n_freq ; i++ ) {
        Vectorcf temp = kernel.row(i).array();
        kernel.row(i) = fft.fwd(temp);
    }

    std::vector<Eigen::Triplet<std::complex<float>>> tripletList;

    // Set small values to zero and non-small values to conjugate of itself / fft_window_size   
    for ( int i = 0 ; i < params.n_freq ; i++ ) {
        for ( int j = 0 ; j < params.fft_window_size ; j++ ) {
            if ( kernel.cwiseAbs().array()(i, j) < 1e-2 ) {
                // do nothing
                ;
            }
            else {
                tripletList.push_back(Eigen::Triplet<std::complex<float>>(i, j,
                    std::complex<float>(1.0f/params.fft_window_size, 0.0f) *
                    conj(kernel(i, j))
                ));
            }
        }
    }
    _kernel.setFromTriplets(tripletList.begin(), tripletList.end());
    _kernel.makeCompressed();
}

py::array_t<float> CQ::compute_cqt(py::array_t<float> audio) {
    std::cout<<"CQ::compute_cqt()"<<std::endl;
    py::buffer_info buf = audio.request();
    int length = buf.shape[0];  
    int n_channels = buf.shape[1];  
    std::cout<<"length: "<<length<<std::endl;
    std::cout<<"n_channels: "<<n_channels<<std::endl;
    // TODO
    //temporarily return the input audio
    py::array_t<float> result = py::array_t<float>(audio);
    return result;
}
