#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
// #include "BasiCPP_Pitch.h"
#include "CQT.h"
// namespace py = pybind11;
PYBIND11_MODULE(BasiCPP_Pitch, m) {
    // py::class<BasiCPP_Pitch>(m, "BasiCPP_Pitch")
    //     .def(py::init<>())
    //     .def("transcribeAudio", &BasiCPP_Pitch::transcribeAudio);
    py::class_<CQParams>(m, "CQParams")
        .def(py::init<float, int, float, float, int>(), py::arg("sample_rate"), py::arg("bins_per_octave"), py::arg("freq_min"), py::arg("freq_max"), py::arg("n_bins"))
        .def_readwrite("sample_rate", &CQParams::sample_rate)
        .def_readwrite("bins_per_octave", &CQParams::bins_per_octave)
        .def_readwrite("freq_min", &CQParams::freq_min)
        .def_readwrite("freq_max", &CQParams::freq_max)
        .def_readwrite("n_bins", &CQParams::n_bins);
    m.def("CQT", &constantQTransform);
}