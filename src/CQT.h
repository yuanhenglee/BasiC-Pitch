#pragma once

#include "typedef.h"
#include <vector>

class CQParams {
    public:
        // init params  
        int sample_rate; // == sr in basic_pitch
        int bins_per_octave;
        int n_bins; 
        float freq_min; // == fmin in basic_pitch
        int sample_per_frame; // == hop_length in basic_pitch

        // computed params
        float freq_max; // == fmax in basic_pitch
        float quality_factor; // == Q in basic_pitch
        int fft_window_size; // 
        int frame_per_second;
        int n_octaves; // == n_octaves in basic_pitch
        float fmin_t; // == fmin_t in basic_pitch, lowest frequency bin for the top octave kernel
        float fmax_t; // == fmax_t in basic_pitch

        // default init params
        float downsample_factor = 1.0f;

        CQParams( bool contour );

//         CQParams(
//             int sample_rate,
//             int bins_per_octave,
//             int n_bins,
//             float freq_min,
//             int hop
//         );
};

class CQ {
    public:
        CQ();
        ~CQ();

        // compute cqt API for Eigen IO
        // Matrixf cqtEigen(const Vectorf& x);
        Matrixf computeCQT(const Vectorf& x, bool batch_norm);

        // Return the cqt feature with harmonic stacking, for vector of matrix IO
        VecMatrixf cqtHarmonic(const Vectorf& x, bool batch_norm);

        // get the kernel matrix, just for testing
        Matrixcf getKernel();

        // get the lowpass filter, just for testing
        Vectorf getFilter();

    private:
        
        // the kernel matrix
        // Eigen::SparseMatrix<std::complex<float>> _kernel;
        Matrixcf _kernel;
        
        CQParams params;

        // for the normalization
        Vectorcf _lengths;

        // lowpass filter for downsampling
        Vectorf _filter_kernel;

        // compute the kernel matrix
        void computeKernel();

        // compute the cqt for input audio
        Matrixcf forward( const Vectorf& x, int hop_length );

        // harmonic stacking
        VecMatrixf harmonicStacking(const Matrixf& cqt , int bins_per_semitone, std::vector<float> harmonics, int n_output_freqs);
};  
