#include "CQT.h"
#include "constant.h"

#include <iostream>
#include <cassert>
#include <cmath>
#include <chrono>

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

CQParams::CQParams( bool contour ) {
    sample_rate = SAMPLE_RATE;
    bins_per_octave = SEMITON_PER_OCTAVE * 
        (contour ? CONTOURS_BINS_PER_SEMITONE : NOTES_BINS_PER_SEMITONE);
    n_bins = N_SEMITONES * 
        (contour ? CONTOURS_BINS_PER_SEMITONE : NOTES_BINS_PER_SEMITONE);
    freq_min = MIN_FREQ;
    sample_per_frame = FFT_HOP;
 

    n_octaves = static_cast<int>(ceil(static_cast<float>(n_bins) / bins_per_octave));
    fmin_t = freq_min * pow(2.0f, n_octaves - 1.0f);
    if ( n_bins % bins_per_octave == 0 ) {
        fmax_t = fmin_t * pow(2.0f, 1.0f - 1.0f/ bins_per_octave);
    }
    else {
        fmax_t = fmin_t * pow(2.0f, static_cast<float>(n_bins % bins_per_octave - 1.0f) / bins_per_octave);
    }
    fmin_t = fmax_t / pow(2.0f, 1 - 1.0f / bins_per_octave);

    freq_max = freq_min * pow(2.0f, static_cast<float>(n_bins-1) / bins_per_octave);
    quality_factor = 1.0f / (pow(2.0f, 1.0f / bins_per_octave) - 1.0f);
    fft_window_size = static_cast<int>(pow(2.0f, 
        ceil(
            log2(quality_factor * sample_rate / fmin_t) + 1e-6
        )));
    frame_per_second = static_cast<int>(static_cast<float>(sample_rate) / sample_per_frame);
}

CQ::CQ(CQParams params) : params(params) {
    computeKernel();
}

CQ::~CQ() = default;

void CQ::computeKernel() {
    int _n_bins = std::min(params.bins_per_octave, params.n_bins);
    Matrixcf kernel = Matrixcf::Zero(_n_bins, params.fft_window_size);
    _lengths = Vectorf::Zero(_n_bins);
    
    Eigen::FFT<float> fft;

    // Calculate the kernel for each frequency bin
    for ( int i = 0 ; i < _n_bins ; i++ ) {

        float freq = params.fmin_t * pow(2.0f, static_cast<float>(i) / params.bins_per_octave);

        int _l = ceil(params.quality_factor * params.sample_rate / freq);
        _lengths[i] = _l;
        
        int start = static_cast<int>(ceil( params.fft_window_size / 2.0f - _l / 2.0f)) - (_l%2);
        
        Vectorcf temporal_kernel = Vectorcf::Zero(_l);

        Vectorcf fft_window = getHann(_l);

        for ( int j = 0 ; j < _l ; j++ ) {
            temporal_kernel[j] = fft_window[j];
            std::complex<float> temp = std::complex<float>(
                0.0f, 2.0f * M_PI * (j - (_l-1)/2 ) * freq / params.sample_rate
            );
            temporal_kernel[j] *= std::exp(temp);
            temporal_kernel[j] /= _l;
        }

        kernel.block(i, start, 1, _l) = temporal_kernel;
    }

    // for ( int i = 0 ; i < _n_bins ; i++ ) {
    //     Vectorcf temp = kernel.row(i).array();
    //     kernel.row(i) = fft.fwd(temp);
    // }

    std::vector<Eigen::Triplet<std::complex<float>>> tripletList;

    // Set small values to zero and non-small values to conjugate of itself / fft_window_size   
    for ( int i = 0 ; i < _n_bins ; i++ ) {
        for ( int j = 0 ; j < params.fft_window_size ; j++ ) {
            if ( kernel.cwiseAbs().array()(i, j) < 0 ) {
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
    _kernel.resize(_n_bins, params.fft_window_size);
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

    Matrixf cqt_feat = forward(audio);

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