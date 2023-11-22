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
        float sample_rate;
        int bins_per_octave;
        float freq_min;
        float freq_max;
        int n_bins;

        // computed params
        float n_freq;
        float quality_factor;
        int fft_window_size;

        CQParams(float sample_rate, int bins_per_octave, float freq_min, float freq_max, int n_bins);   
};

class CQ {
    public:
        CQ(CQParams params);
        ~CQ();
        py::array_t<float> compute_cqt(py::array_t<float> audio);
    private:
        int _n_freq;
        Eigen::SparseMatrix<std::complex<float>> _kernel;
        CQParams params;
        void computeKernel();
        Matrixf forward(Vectorcf x);
};  




py::array_t<float> constantQTransform( py::array_t<float> audio, CQParams params );

#endif // CQT_H
