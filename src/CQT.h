#ifndef CQT_H
#define CQT_H

#include "Eigen/Core"
#include "Eigen/Sparse"
#include "unsupported/Eigen/FFT"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
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
        int freq_min;
        int freq_max;
        // int n_bins; deprecated, replaced by n_freq
        float hop_size;

        // computed params
        int n_freq; // n_bins
        float quality_factor;
        int fft_window_size;
        int frame_per_second;
        int sample_per_frame;

        CQParams(
            int sample_rate,
            int bins_per_octave,
            int freq_min,
            int freq_max,
            float hop_size
        );
};

class CQ {
    public:
        CQ(CQParams params);
        ~CQ();

        // compute cqt API for POD IO
        void cqt_POD(float* audio, int &audio_len, float* output, int &output_len);        

        // compute cqt API for np.array IO
        py::array_t<float> cqt_Py(py::array_t<float> audio);

    private:
        int _n_freq;
        Eigen::SparseMatrix<std::complex<float>> _kernel;
        CQParams params;
        void computeKernel();
        Matrixf forward( const Vectorf& x);
};  




py::array_t<float> constantQTransform( py::array_t<float> audio, CQParams params );

#endif // CQT_H
