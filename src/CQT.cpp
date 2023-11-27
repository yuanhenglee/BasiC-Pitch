#include "CQT.h"

#include <iostream>
#include <cassert>
#include <chrono>

static std::vector<float> getHammingWindow(int window_size) {
    std::vector<float> window(window_size);
    for ( int i = 0 ; i < window_size ; i++ ) {
        window[i] = 0.54f - 0.46f * cos(2.0f * M_PI * i / (window_size - 1));
    }
    return window;
}

CQParams::CQParams(
    int sample_rate,
    int bins_per_octave,
    int n_bins,
    float freq_min,
    int hop
)
    : sample_rate(sample_rate), bins_per_octave(bins_per_octave), freq_min(freq_min), hop(hop) {  

    freq_max = freq_min * pow(2.0f, static_cast<float>(n_bins-1) / bins_per_octave);
    n_freq = n_bins;
    quality_factor = 1.0f / (pow(2.0f, 1.0f / bins_per_octave) - 1.0f);
    fft_window_size = static_cast<int>(pow(2.0f, 
        ceil(
            log2(quality_factor * sample_rate / freq_min) + 1e-6
        )));
    // frame_per_second = static_cast<int>(1 / hop_size);
    // sample_per_frame = round(static_cast<float>(sample_rate) / frame_per_second);
    sample_per_frame = hop;
    frame_per_second = static_cast<int>(static_cast<float>(sample_rate) / sample_per_frame);
}   

CQ::CQ(CQParams params) : params(params) {
    // _kernel = Eigen::SparseMatrix<std::complex<float>>(params.n_freq, params.fft_window_size);
    _kernel.resize(params.n_freq, params.fft_window_size);
    computeKernel();
}

CQ::~CQ() = default;

void CQ::computeKernel() {
    Matrixcf kernel = Matrixcf::Zero(params.n_freq, params.fft_window_size);
    _lengths = Vectorf::Zero(params.n_freq);
    
    // Calculate the kernel for each frequency bin
    for ( int i = 0 ; i < params.n_freq ; i++ ) {

        float freq = params.freq_min * pow(2.0f, static_cast<float>(i) / params.bins_per_octave);

        _lengths[i] = ceil(params.quality_factor * params.sample_rate / freq);

        int window_length = 2 * round(
            params.quality_factor * params.sample_rate / freq / 2.0f
        ) + 1;


        
        Vectorcf temporal_kernel = Vectorcf::Zero(window_length);

        std::vector<float> hamming_window = getHammingWindow(window_length);
        // std::vector<float> arange_window = getArangeWindow(window_length);

        for ( int j = 0 ; j < window_length ; j++ ) {
            std::complex<float> temp = std::complex<float>(
                0.0f, 2.0f * M_PI * params.quality_factor * (j - (window_length-1)/2 ) / static_cast<float>(window_length)
            );
            temporal_kernel[j] =
                std::complex<float>(hamming_window[j] / window_length, 0.0f)
                * std::exp(temp);
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
            if ( kernel.cwiseAbs().array()(i, j) < 1e-1 ) {
                // do nothing
                ;
            }
            else {
                tripletList.emplace_back(
                    Eigen::Triplet<std::complex<float>>(i, j,
                    std::complex<float>(1.0f/params.fft_window_size, 0.0f) *
                    conj(kernel(i, j))
                ));
            }
        }
    }
    _kernel.setFromTriplets(tripletList.begin(), tripletList.end());
    _kernel.makeCompressed();
}

Matrixf CQ::forward( const Vectorf &x ) {
    // number of fft windows we can do is the number of frames we can divide from x plus 1
    int n_fft_x = x.size() / params.sample_per_frame;
    assert(n_fft_x > 0 && "Input signal is too short");
    int left = ceil((params.fft_window_size - params.sample_per_frame) / 2.0f);
    int right = floor((params.fft_window_size - params.sample_per_frame) / 2.0f);

    Vectorf padded_x = Vectorf::Zero(left + x.size() + right);
    padded_x.segment(left, x.size()) = x;

    Eigen::FFT<float> fft;
    Matrixcf fft_x(n_fft_x, params.fft_window_size );
    for ( int i = 0 , j = 0 ; i < n_fft_x ; i++, j += params.sample_per_frame ) {
        Vectorf frame = padded_x.segment(j, params.fft_window_size).array();
        fft_x.row(i) = fft.fwd(frame);
    }
    
    Matrixcf kernel_output = _kernel * fft_x.transpose();
    Matrixf cqt_feat = kernel_output.cwiseAbs();

    // librosa style normalization
    // cqt_feat *= tf.math.sqrt(tf.cast(self.lengths.reshape((-1, 1, 1)), self.dtype))
    // cqt_feat = cqt_feat.array().colwise() * _lengths.transpose().array().sqrt();

    return cqt_feat;
}

void CQ::cqtPOD(float* audio, int &audio_len, float* output) {

    Eigen::VectorXf x = Eigen::Map<Eigen::VectorXf>(audio, audio_len);
    Eigen::MatrixXf cqt_feat = forward(x);

    assert(cqt_feat.rows() == params.n_freq && "CQT feature rows not equal to n_freq");

    for ( int i = 0 ; i < cqt_feat.rows() ; i++ ) {
        for ( int j = 0 ; j < cqt_feat.cols() ; j++ ) {
            output[i * cqt_feat.cols() + j] = cqt_feat.coeff(i, j);
        }
    }
}

py::array_t<float> CQ::cqtPy(py::array_t<float> audio) {
    // NOTE : input audio should be 1D array at this point
    py::buffer_info buf = audio.request();
    int audio_len = buf.shape[0];

    Eigen::VectorXf x = Eigen::Map<Eigen::VectorXf>((float*)buf.ptr, audio_len);
    Eigen::MatrixXf cqt_feat = forward(x);

    int n_fft_x = audio_len / params.hop;
    py::array_t<float> result = py::array_t<float>({params.n_freq, n_fft_x});
    auto r = result.mutable_unchecked<2>();
    for ( int i = 0 ; i < params.n_freq ; i++ ) {
        for ( int j = 0 ; j < n_fft_x ; j++ ) {
            r(i, j) = cqt_feat.coeff(i, j);
        }
    }

    return result;
}

Matrixf CQ::cqtEigen(const Vectorf& audio) {
    // NOTE : input audio should be 1D array at this point
    
    // log wall time
    // auto start = std::chrono::high_resolution_clock::now();

    Eigen::MatrixXf cqt_feat = forward(audio);

    // auto end = std::chrono::high_resolution_clock::now();
    // std::cout<< "cqt run time"<< 
    //     std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
    //     << std::endl; 
    return cqt_feat;
}

Matrixf CQ::cqtEigenHarmonic(const Vectorf& audio) {

    Eigen::MatrixXf cqt_feat = forward(audio);
    // TODO : implement harmonic stacking
    
    return cqt_feat;
}


py::array_t<std::complex<float>> CQ::getKernel() {
    py::array_t<std::complex<float>> result = py::array_t<std::complex<float>, py::array::c_style>(
        {_kernel.rows(), _kernel.cols()}
    );
    py::buffer_info buf = result.request();
    std::complex<float>* res_ptr = (std::complex<float>*)buf.ptr;

    for ( int i = 0 ; i < _kernel.rows() ; i++ ) {
        for ( int j = 0 ; j < _kernel.cols() ; j++ ) {
            res_ptr[i * _kernel.cols() + j] = _kernel.coeff(i, j);
        }
    }

    return result;
}