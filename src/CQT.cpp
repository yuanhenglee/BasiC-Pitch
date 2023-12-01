#include "CQT.h"
#include "constant.h"
#include "utils.h"

#include <iostream>
#include <cassert>
#include <cmath>
#include <chrono>

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
    // computeKernel();
    
    // Compute the length of the kernels for later normalization
    int _n_bins = params.n_bins;
    _lengths = Vectorcf::Zero(_n_bins);
    for ( int i = 0 ; i < _n_bins ; i++ ) {
        float freq = params.freq_min * pow(2.0f, static_cast<float>(i) / params.bins_per_octave);
        float _l = ceil(params.quality_factor * params.sample_rate / freq);
        std::complex<float> _l_sqrt_complex = std::complex<float>(sqrt(_l), 0.0f);
        _lengths[i] = _l_sqrt_complex;
    }

    loadDefaultKernel(_kernel);
}

CQ::~CQ() = default;

// decrepated, load precomputed kernel instead
void CQ::computeKernel() {
    int _n_bins = std::min(params.bins_per_octave, params.n_bins);
    _kernel = Matrixcf::Zero(_n_bins, params.fft_window_size);
    _lengths = Vectorf::Zero(_n_bins);
    
    Eigen::FFT<float> fft;

    // Calculate the _kernel for each frequency bin
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

        _kernel.block(i, start, 1, _l) = temporal_kernel;
    }

    for ( int i = 0 ; i < _n_bins ; i++ ) {
        Vectorcf temp = _kernel.row(i).array();
        _kernel.row(i) = fft.fwd(temp);
    }
}

Matrixcf CQ::forward( const Vectorf &x, int hop_length ) {

    // due to the reflection padding, the output size plus 1
    int n_fft_x = x.size() / hop_length + 1;
    Vectorf padded_x = reflectionPadding(x, params.fft_window_size / 2);
    Matrixcf cqt_feat = Matrixcf::Zero(n_fft_x, _kernel.rows());

    for ( int i = 0 ; i < n_fft_x ; i++ ) {
        int start = i * hop_length;
        Vectorcf temp = padded_x.segment(start, params.fft_window_size).cast<std::complex<float>>();
        cqt_feat.row(i) = temp * _kernel.transpose();
    }

    return cqt_feat.transpose();
}

// std::vector<Matrixf>
py::array_t<float>
CQ::harmonicStacking(const Matrixcf& cqt , int bins_per_semitone, std::vector<float> harmonics, int n_output_freqs) {
    
    // std::vector<Matrixf> stacking_features(harmonics.size(), Matrixf::Zero(cqt.rows(), cqt.cols()));

    // n_output_freqs = std::min(n_output_freqs, static_cast<int>(cqt.rows()));

    std::vector<size_t> shape = {
        harmonics.size(),
        static_cast<unsigned long>(n_output_freqs),
        static_cast<unsigned long>(cqt.cols())
    };
    py::array_t<float> result(shape);
    // py::buffer_info buf = result.request();
    // float* res_ptr = (float*)buf.ptr;
    // int n_bins = cqt.rows(), n_frames = cqt.cols();

    // for ( size_t i = 0 ; i < harmonics.size() ; i++ ) {
        
    //     Matrixf padded = Matrixf::Zero(n_bins, n_frames);
    //     int shift = static_cast<int>(round(12.0f * bins_per_semitone * log2(harmonics[i])));

    //     if (shift == 0)
    //         padded = cqt;
    //     else if (shift > 0) {
    //         padded.block(0, 0, n_bins - shift, n_frames) = cqt.block(shift, 0, n_bins - shift, n_frames);
    //     }
    //     else {
    //         padded.block(-shift, 0, n_bins + shift, n_frames) = cqt.block(0, 0, n_bins + shift, n_frames);
    //     }

    //     for ( size_t j = 0 ; j < static_cast<size_t>(n_output_freqs) ; j++ ) {
    //         for ( size_t k = 0 ; k < static_cast<size_t>(n_frames) ; k++ ) {
    //             res_ptr[i * n_output_freqs * n_frames + j * n_frames + k] = padded(j, k);
    //         }
    //     }

    // }

    return result;
}


Matrixf CQ::cqtEigen(const Vectorf& audio) {
    // NOTE : input audio should be 1D array at this point
    int hop = params.sample_per_frame;
    int n_fft_x = audio.size() / hop + 1;
    int _n_bins = _kernel.rows();

    Matrixcf cqt_feat(params.n_bins, n_fft_x );

    // Getting the top octave CQT
    int start = params.n_bins - _n_bins;
    cqt_feat.block(start , 0, _n_bins, n_fft_x) = forward(audio, hop);

    Vectorf audio_down = audio;

    Vectorf filter_kernel = defaultLowPassFilter();

    for ( int i = 1 ; i < params.n_octaves ; i++ ) {
        start -= _n_bins;
        hop /= 2;
        audio_down = downsamplingByN(audio_down, filter_kernel, 2.0f);
        if (start >= 0)
            cqt_feat.block(start, 0, _n_bins, n_fft_x) = forward(audio_down, hop);
        else
            cqt_feat.block(0, 0, _n_bins + start, n_fft_x) = forward(audio_down, hop).block(-start, 0, _n_bins + start, n_fft_x);
    }

    // normalization
    // top_cqt_feat *= params.downsample_factor; // we don't need this since the factor is 1
    // librosa fasion normalization
    cqt_feat = cqt_feat.array() * _lengths.transpose().array().replicate(1, n_fft_x);

    // store only the magnitude
    Matrixf res = cqt_feat.cwiseAbs();

    return res;
}

// Matrixf CQ::cqtEigenHarmonic(const Vectorf& audio) {
py::array_t<float> CQ::cqtEigenHarmonic(const Vectorf& audio) {

    Matrixcf cqt_feat = forward(audio, params.sample_per_frame);

    // std::vector<float> harmonics = {0.5};
    // for ( int i = 1 ; i < N_HARMONICS ; i++ ) {
    //     harmonics.emplace_back(i);
    // }

    // return harmonicStacking(
    //     cqt_feat,
    //     CONTOURS_BINS_PER_SEMITONE,
    //     harmonics,
    //     N_BINS_CONTOUR
    // );
    return py::array_t<float>();
}


Matrixcf CQ::getKernel() {
    return _kernel;
}