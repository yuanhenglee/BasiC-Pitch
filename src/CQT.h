#ifndef CQT_H
#define CQT_H

#include "Eigen/Core"
#include "Eigen/Sparse"
#include "unsupported/Eigen/FFT"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <cmath>

typedef Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor> Vectorf;
typedef Eigen::Matrix<std::complex<float>, 1, Eigen::Dynamic, Eigen::RowMajor>
    Vectorcf;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    Matrixf;
typedef Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    Matrixcf;

namespace py = pybind11;

class CQParams {
    public:
        // init params  
        int sample_rate;
        int bins_per_octave;
        int n_bins; 
        float freq_min;
        int hop; // sample_per_frame

        // computed params
        float freq_max;
        int n_freq; // n_bins
        float quality_factor;
        int fft_window_size;
        int sample_per_frame;
        int frame_per_second;

        CQParams(
            int sample_rate,
            int bins_per_octave,
            int n_bins,
            float freq_min,
            int hop
        );
};

class CQ {
    public:
        CQ(CQParams params);
        ~CQ();

        // compute cqt API for POD IO
        void cqtPOD(float* audio, int &audio_len, float* output);        

        // compute cqt API for np.array IO
        py::array_t<float> cqtPy(py::array_t<float> audio);

        // compute cqt API for Eigen IO
        Matrixf cqtEigen(const Vectorf& x);

        // get the kernel matrix, just for testing
        py::array_t<std::complex<float>> getKernel();

    private:
        
        // the kernel matrix
        Eigen::SparseMatrix<std::complex<float>> _kernel;
        
        CQParams params;

        // for the normalization
        Vectorf _lengths;

        // compute the kernel matrix
        void computeKernel();

        // compute the cqt for input audio
        Matrixf forward( const Vectorf& x);
};  


#endif // CQT_H
