#include "CQT.h"
#include "constant.h"
#include "utils.h"
#include "loader.h"

#include <iostream>
#include <cassert>
#include <cmath>

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

CQ::CQ() : params(CQParams(true)) {
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
    loadDefaultLowPassFilter(_filter_kernel);
}

CQ::~CQ() = default;

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

// output shape = (n_harmonics, n_frames, n_bins)
VecMatrixf CQ::harmonicStacking(const Matrixf& cqt , int bins_per_semitone, std::vector<float> harmonics, int n_output_freqs) {
    
    int n_bins = cqt.rows(), n_frames = cqt.cols();

    VecMatrixf result(harmonics.size());
    for ( size_t i = 0 ; i < harmonics.size() ; i++ ) {
        
        Matrixf padded = Matrixf::Zero(n_bins, n_frames);
        int shift = static_cast<int>(round(12.0f * bins_per_semitone * log2(harmonics[i])));

        if (shift == 0)
            padded = cqt;
        else if (shift > 0) {
            padded.block(0, 0, n_bins - shift, n_frames) = cqt.block(shift, 0, n_bins - shift, n_frames);
        }
        else {
            padded.block(-shift, 0, n_bins + shift, n_frames) = cqt.block(0, 0, n_bins + shift, n_frames);
        }

        Matrixf temp = padded.block(0, 0, n_output_freqs, n_frames);

        result[i] = temp.transpose();
    }

    return result;
}


// Matrixf CQ::cqtEigen(const Vectorf& audio) {
Matrixf CQ::computeCQT(const Vectorf& audio, bool batch_norm) {
    // NOTE : input audio should be 1D array at this point
    int hop = params.sample_per_frame;
    int n_fft_x = audio.size() / hop + 1;
    int _n_bins = _kernel.rows();

    Matrixcf cqt_feat(params.n_bins, n_fft_x );

    // Getting the top octave CQT
    int start = params.n_bins - _n_bins;
    cqt_feat.block(start , 0, _n_bins, n_fft_x) = forward(audio, hop);

    Vectorf audio_down = audio;

    for ( int i = 1 ; i < params.n_octaves ; i++ ) {
        start -= _n_bins;
        hop /= 2;
        audio_down = downsamplingByN(audio_down, _filter_kernel, 2.0f);
        if (start >= 0)
            cqt_feat.block(start, 0, _n_bins, n_fft_x) = forward(audio_down, hop);
        else
            cqt_feat.block(0, 0, _n_bins + start, n_fft_x) = forward(audio_down, hop).block(-start, 0, _n_bins + start, n_fft_x);
    }

    // normalization
    // top_cqt_feat *= params.downsample_factor; // we don't need this since the factor is 1
    // librosa fasion normalization
    cqt_feat = cqt_feat.array() * _lengths.transpose().array().replicate(1, n_fft_x);

    // power of magnitude
    Matrixf power = cqt_feat.array().cwiseAbs2() + 1e-10;
    Matrixf log_power = 10.0f * power.array().log10();
    log_power = (log_power.array() - log_power.minCoeff()) / (log_power.maxCoeff() - log_power.minCoeff());

    // batch normalization
    if ( batch_norm) {
        constexpr float gamma = 0.48823851346969604;
        constexpr float beta = 0.3687160313129425;
        constexpr float mean = 0.5021218657493591;
        constexpr float var = 0.03773479163646698;

        log_power = (log_power.array() - mean) * gamma / sqrt(var + 0.001f) + beta;
    }

    return log_power;
}

// Matrixf CQ::cqtEigenHarmonic(const Vectorf& audio) {
VecMatrixf CQ::cqtHarmonic(const Vectorf& audio, bool batch_norm) {

    // Matrixf cqt_feat = cqtEigen(audio);
    Matrixf cqt_feat = computeCQT(audio, batch_norm);

    std::vector<float> harmonics = {0.5};
    for ( int i = 1 ; i < N_HARMONICS ; i++ ) {
        harmonics.emplace_back(i);
    }

    VecMatrixf hs = harmonicStacking(
        cqt_feat,
        CONTOURS_BINS_PER_SEMITONE,
        harmonics,
        N_BINS_CONTOUR
    );
    return hs;
}

Matrixcf CQ::getKernel() {
    return _kernel;
}

Vectorf CQ::getFilter() {
    return _filter_kernel;
}