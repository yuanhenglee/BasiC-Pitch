#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "CQT.h"
#include "constant.h"

PYBIND11_MODULE(BasiCPP_Pitch, m) {
    // py::class<BasiCPP_Pitch>(m, "BasiCPP_Pitch")
    //     .def(py::init<>())
    //     .def("transcribeAudio", &BasiCPP_Pitch::transcribeAudio);

    py::class_<CQParams>(m, "CQParams")
        .def(py::init<bool>(), py::arg("contour") = true)
        .def("__repr__",
        [] (const CQParams &params) {
            return "<CQParams\n"
                "sample_rate: " + std::to_string(params.sample_rate) + "\n"
                "bins_per_octave: " + std::to_string(params.bins_per_octave) + "\n"
                "n_bins: " + std::to_string(params.n_bins) + "\n"
                "freq_min: " + std::to_string(params.freq_min) + "\n"
                "sample_per_frame: " + std::to_string(params.sample_per_frame) + "\n"
                "freq_max: " + std::to_string(params.freq_max) + "\n"
                "quality_factor: " + std::to_string(params.quality_factor) + "\n"
                "fft_window_size: " + std::to_string(params.fft_window_size) + "\n"
                "frame_per_second: " + std::to_string(params.frame_per_second) + "\n"
                "n_octaves: " + std::to_string(params.n_octaves) + "\n"
                "fmin_t: " + std::to_string(params.fmin_t) + "\n"
                "fmax_t: " + std::to_string(params.fmax_t) + "\n"
                ">";
        });
    py::class_<CQ>(m, "CQ")
        .def(py::init<CQParams>(), py::arg("params"))
        // .def("computeCQT", &CQ::cqtPy)
        // .def("computeCQT", &CQ::cqtEigen)
        .def("computeCQT", &CQ::cqtEigenHarmonic)
        .def("getKernel", &CQ::getKernel);

}