#include "typedef.h"
#include "CQT.h"
#include "utils.h"
#include "layer.h"
#include "cnn.h"
#include "amtModel.h"

void bind_layer( py::module &m ) {
    auto m_layer = m.def_submodule("layer");
    py::class_<Layer>(m_layer, "Layer");
    py::class_<Conv2D, Layer>(m_layer, "Conv2D")
        .def("getWeights", [] ( const Conv2D &conv2d ) {
            VecVecMatrixf weights = conv2d.getWeights();
            return weights[0][0];
        })
        .def("forward", [] ( const Conv2D &conv2d, py::array_t<float> input ) {
            VecMatrixf input_tensor = pyarray2mat3D(input);
            VecMatrixf output_tensor = conv2d.forward(input_tensor);
            return mat3D2pyarray(output_tensor);
        });
    py::class_<ReLU, Layer>(m_layer, "ReLU")
        .def(py::init<>())
        .def("forward", [] ( const ReLU &relu, py::array_t<float> input ) {
            VecMatrixf input_tensor = pyarray2mat3D(input);
            VecMatrixf output_tensor = relu.forward(input_tensor);
            return mat3D2pyarray(output_tensor);
        });
    py::class_<Sigmoid, Layer>(m_layer, "Sigmoid")
        .def(py::init<>())
        .def("forward", [] ( const Sigmoid &sigmoid, py::array_t<float> input ) {
            VecMatrixf input_tensor = pyarray2mat3D(input);
            VecMatrixf output_tensor = sigmoid.forward(input_tensor);
            return mat3D2pyarray(output_tensor);
        });
}

void bind_cnn( py::module &m ) {
    py::class_<CNN>(m, "CNN")
        .def(py::init<std::string>())
        .def("__repr__",
        [] (const CNN &cnn) {
            return cnn.get_name();
        })
        .def("forward", [] ( const CNN &cnn, py::array_t<float> input ) {
            VecMatrixf input_tensor = pyarray2mat3D(input);
            VecMatrixf output_tensor = cnn.forward(input_tensor);
            return mat3D2pyarray(output_tensor);
        })
        .def("getFirstKernel", [] ( const CNN &cnn ) {
            std::vector<Layer*> layers(cnn.get_layers());
            Matrixf weights = dynamic_cast<Conv2D*>(layers[0])->getWeights()[0][0];
            return weights;
        });
}

// bind the amtModel class
void bind_amtModel( py::module &m ) {
    py::class_<amtModel>(m, "amtModel")
        .def(py::init<>())
        .def("inference", &amtModel::inference)
        .def("getCQ", &amtModel::getCQ)
        .def("getYo", &amtModel::getYo)
        .def("getYp", &amtModel::getYp)
        .def("getYn", &amtModel::getYn);
}


// bind the utils functions
void bind_utils( py::module &m ) {
    auto m_utils = m.def_submodule("utils");
    m_utils.def("testConv2d", [] ( Matrixf &x, Matrixf &filter_kernel, int stride ) {
        Matrixf output = conv2d(x, filter_kernel, stride);
        return output;
    });
    m_utils.def("testMatConversion", [] ( py::array_t<float> input ) {
        printPyarray(input);
        VecMatrixf input_tensor = pyarray2mat3D(input);
        printVecMatrixf(input_tensor);
        auto output = mat3D2pyarray(input_tensor);
        printPyarray(output);
        return output;
    });
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
        .def("harmonicStacking", [] ( CQ &cq, Vectorf &x ) {
            VecMatrixf output_tensor = cq.cqtHarmonic(x);
            return mat3D2pyarray(output_tensor);
        });
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