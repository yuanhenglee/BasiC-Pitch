#ifndef CQT_H
#define CQT_H

#include "Eigen/Core"
#include "Eigen/Sparse"
#include "unsupported/Eigen/FFT"
#include "unsupported/Eigen/CXX11/Tensor"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <cmath>
#include <vector>

typedef Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor> Vectorf;
typedef Eigen::Matrix<std::complex<float>, 1, Eigen::Dynamic, Eigen::RowMajor>
    Vectorcf;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    Matrixf;
typedef Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    Matrixcf;
typedef Eigen::Tensor<float, 3, Eigen::RowMajor> Tensor3f;
typedef Eigen::Tensor<float, 2, Eigen::RowMajor> Tensor2f;

namespace py = pybind11;

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
        CQ(CQParams params);
        ~CQ();

        // compute cqt API for Eigen IO
        Matrixf cqtEigen(const Vectorf& x);

        // Return the cqt feature with harmonic stacking
        py::array_t<float> cqtEigenHarmonic(const Vectorf& x);

        // get the kernel matrix, just for testing
        // py::array_t<std::complex<float>> getKernel();
        Matrixcf getKernel();

    private:
        
        // the kernel matrix
        // Eigen::SparseMatrix<std::complex<float>> _kernel;
        Matrixcf _kernel;
        
        CQParams params;

        // for the normalization
        Vectorcf _lengths;

        // compute the kernel matrix
        void computeKernel();

        // compute the cqt for input audio
        Matrixcf forward( const Vectorf& x, int hop_length );

        // harmonic stacking
        Tensor3f harmonicStacking(const Matrixf& cqt , int bins_per_semitone, std::vector<float> harmonics, int n_output_freqs);
};  


#endif // CQT_H
