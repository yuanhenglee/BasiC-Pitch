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

#include <cassert>

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

CQParams::CQParams(
    int sample_rate,
    int bins_per_octave,
    int freq_min,
    int freq_max,
    float hop_size
)
    : sample_rate(sample_rate), bins_per_octave(bins_per_octave), freq_min(freq_min), freq_max(freq_max), hop_size(hop_size) {  

    n_freq = round(
        static_cast<float>(bins_per_octave) * log2(
            static_cast<float>(freq_max) / freq_min
    ));
    quality_factor = 1.0f / (pow(2.0f, 1.0f / bins_per_octave) - 1.0f);
    fft_window_size = static_cast<int>(pow(2.0f, 
        ceil(
            log2(quality_factor * sample_rate / freq_min) + 1e-6
        )));
    frame_per_second = static_cast<int>(1 / hop_size);
    sample_per_frame = round(static_cast<float>(sample_rate) / frame_per_second);
}   

CQ::CQ(CQParams params) : params(params) {
    _kernel = Eigen::SparseMatrix<std::complex<float>>(params.n_freq, params.fft_window_size);
    computeKernel();
}

CQ::~CQ() = default;

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
            if ( kernel.cwiseAbs().array()(i, j) < 1e-8 ) {
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
    _kernel.resize(params.n_freq, params.fft_window_size);
    _kernel.setFromTriplets(tripletList.begin(), tripletList.end());
    _kernel.makeCompressed();

    // std::cout<<_kernel<<std::endl;
}

Matrixf CQ::forward( const Vectorf &x ) {
    int n_sample_x = static_cast<float>(x.size()) / params.sample_per_frame;
    assert(n_sample_x > 0 && "Input signal is too short");
    int left = ceil((params.fft_window_size - params.frame_per_second) / 2.0f);
    int right = floor((params.fft_window_size - params.frame_per_second) / 2.0f);

    Vectorf padded_x = Vectorf::Zero(left + x.size() + right);
    padded_x.segment(left, x.size()) = x;

    Eigen::FFT<float> fft;
    Matrixcf fft_x(n_sample_x, params.fft_window_size );
    for ( int i = 0 , j = 0 ; i < n_sample_x ; i++, j += params.sample_per_frame ) {
        Vectorf frame = padded_x.segment(j, params.fft_window_size).array();
        fft_x.row(i) = fft.fwd(frame);
    }
    
    std::cout<<_kernel.row(0)<<std::endl;
    std::cout<<fft_x.row(0)<<std::endl;

    Matrixcf kernel_output = _kernel * fft_x.transpose();
    Matrixf cqt_feat = kernel_output.cwiseAbs().array();

    // cqt_feat = cqt_feat.array() + 1e-9;

    // float ref = cqt_feat.maxCoeff();
    // cqt_feat = 20 * cqt_feat.array().max(1e-10).log10() - 20 * log10(ref);

    return cqt_feat.transpose();
}

void CQ::cqt_POD(float* audio, int &audio_len, float* output, int &output_len) {

    Eigen::VectorXf x = Eigen::Map<Eigen::VectorXf>(audio, audio_len);
    Eigen::MatrixXf cqt_feat = forward(x);

    output = new float[cqt_feat.rows() * cqt_feat.cols()];

    output_len = 0;
    for ( int i = 0 ; i < cqt_feat.rows() ; i++ ) {
        for ( int j = 0 ; j < cqt_feat.cols() ; j++ ) {
            output[output_len++] = cqt_feat(i, j);
        }
    }
}

py::array_t<float> CQ::cqt_Py(py::array_t<float> audio) {
    // NOTE : input audio should be 1D array at this point
    py::buffer_info buf = audio.request();
    int audio_len = buf.shape[0];  

    float* res_ptr;
    int res_len;
    cqt_POD((float*)buf.ptr, audio_len, res_ptr, res_len);

    
    py::array_t<float> result = py::array_t<float>(
        {res_len}, // shape
        {sizeof(float)}, // C-style contiguous strides for double
        res_ptr // the data pointer
    );

    result.resize({params.n_freq, res_len / params.n_freq});

    return result;
}
