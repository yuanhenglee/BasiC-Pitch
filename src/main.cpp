#include "typedef.h"
#include "CQT.h"
#include "utils.h"
#include "layer.h"
#include "cnn.h"
#include "amtModel.h"

void bind_layer( py::module &m ) {
    py::class_<Conv2D>(m, "Conv2D")
        .def(py::init<int, int, int, int, int, int>())
        // .def("get_name", &Conv2D::get_name)
        .def("__repr__",
        [] (const Conv2D &layer) {
            return layer.get_name();
        })
        .def("forward", &Conv2D::forward);
}

void bind_cnn( py::module &m ) {
    py::class_<CNN>(m, "CNN")
        .def(py::init<std::string>())
        .def("__repr__",
        [] (const CNN &cnn) {
            return cnn.get_name();
        });
}

// bind the amtModel class
void bind_amtModel( py::module &m ) {
    py::class_<amtModel>(m, "amtModel")
        .def(py::init<>())
        .def("inference", &amtModel::inference)
        .def("getCQ", &amtModel::getCQ);
}


// bind the utils functions
void bind_utils( py::module &m ) {
    // m.def("printMat", &printMat);
    m.def("getHamming", &getHamming);
    m.def("getHann", &getHann);
    m.def("updateEDParams", &updateEDParams);
    m.def("downsamplingByN", &downsamplingByN);
}

// bind the CQParams class
void bind_CQParams( py::module &m ) {
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
                "downsample_factor: " + std::to_string(params.downsample_factor) + "\n"
                ">";
        });
}

// bind the CQ class
void bind_CQ( py::module &m ) {
    py::class_<CQ>(m, "CQ")
        .def(py::init<>())
        .def("computeCQT", &CQ::cqtEigen)
        .def("getKernel", &CQ::getKernel)
        .def("getFilter", &CQ::getFilter)
        .def("harmonicStacking", &CQ::cqtHarmonicPy);
}

PYBIND11_MODULE(BasiCPP_Pitch, m) {
    m.doc() = "BasiCPP_Pitch: A C++ implementation of the pitch detection algorithm";
    bind_layer(m);
    bind_cnn(m);
    bind_amtModel(m);
    bind_utils(m);
    bind_CQParams(m);
    bind_CQ(m);
}