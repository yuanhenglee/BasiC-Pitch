#ifndef CQT_H
#define CQT_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "Eigen/Core"
#include "Eigen/Sparse"
#include "unsupported/Eigen/FFT"
#include <cmath>

namespace py = pybind11;

class CQParams {
    public:
        float sample_rate;
        int bins_per_octave;
        float freq_min;
        float freq_max;
        int n_bins;
        CQParams(float sample_rate, int bins_per_octave, float freq_min, float freq_max, int n_bins);   
};

class CQ {
    public:
        CQ(CQParams params);
        ~CQ();
        py::array_t<float> compute_cqt(py::array_t<float> audio); 
    private:
        void runKernel();
        CQParams params;
};  

py::array_t<float> constantQTransform( py::array_t<float> audio, CQParams params );

#endif // CQT_H
