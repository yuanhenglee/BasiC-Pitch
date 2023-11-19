#ifndef CQT_H
#define CQT_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
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

py::array_t<double> constantQTransform( py::array_t<double> audio, CQParams params );

#endif // CQT_H
