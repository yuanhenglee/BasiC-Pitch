#include "typedef.h"
#include "CQT.h"
#include "utils.h"
#include "nnUtils.h"
#include "layer.h"
#include "cnn.h"
#include "amtModel.h"
#include "note.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <vector>
#include <iostream>
#include <string>

namespace py = pybind11;

// binding utils functions
void printPyarray(py::array_t<float> &pyarray) {
    if ( pyarray.ndim() == 1 ) {
        auto r = pyarray.unchecked<1>();
        for ( int i = 0 ; i < r.shape(0) ; i++ ) {
            std::cout << r(i) << " ";
        }
        std::cout << std::endl;
    }
    else if ( pyarray.ndim() == 2 ) {
        auto r = pyarray.unchecked<2>();
        for ( int i = 0 ; i < r.shape(0) ; i++ ) {
            for ( int j = 0 ; j < r.shape(1) ; j++ ) {
                std::cout << r(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }
    else if ( pyarray.ndim() == 3 ) {
        auto r = pyarray.unchecked<3>();
        for ( int i = 0 ; i < r.shape(0) ; i++ ) {
            std::cout << "i = " << i << std::endl;
            for ( int j = 0 ; j < r.shape(1) ; j++ ) {
                for ( int k = 0 ; k < r.shape(2) ; k++ ) {
                    std::cout << r(i, j, k) << " ";
                }
                std::cout << std::endl;
            }
        }
    }
    else {
        std::cout << "ndim > 3" << std::endl;
    }
}

py::array_t<float> mat3D2pyarray(VecMatrixf &tensor) {
    std::vector<long int> shape = {tensor.size(), tensor[0].rows(), tensor[0].cols()};
    py::array_t<float> pyarray(shape);
    auto r = pyarray.mutable_unchecked<3>();
    for ( py::ssize_t i = 0 ; i < r.shape(0) ; i++ ) {
        for ( py::ssize_t j = 0 ; j < r.shape(1) ; j++ ) {
            for ( py::ssize_t k = 0 ; k < r.shape(2) ; k++ ) {
                r(i, j, k) = tensor[i](j, k);
            }
        }
    }
    return pyarray;
}

VecMatrixf pyarray2mat3D(py::array_t<float> &pyarray) {
    auto r = pyarray.unchecked<3>();
    VecMatrixf tensor(r.shape(0), Matrixf::Zero(r.shape(1), r.shape(2)));
    for ( int i = 0 ; i < r.shape(0) ; i++ ) {
        for ( int j = 0 ; j < r.shape(1) ; j++ ) {
            for ( int k = 0 ; k < r.shape(2) ; k++ ) {
                tensor[i](j, k) = r(i, j, k);
            }
        }
    }
    return tensor;
}

void bind_nnUtils( py::module &m ) {
    auto m_nnUtils = m.def_submodule("nnUtils");
    m_nnUtils.def("im2col", &im2col);
    m_nnUtils.def("col2im", &col2im);
    m_nnUtils.def("testConv2d", [] ( Matrixf &x, Matrixf &filter_kernel, int stride ) {
        Matrixf output = conv2d(x, filter_kernel, stride);
        return output;
    });
}

void bind_note( py::module &m ) {
    auto m_note = m.def_submodule("note");
    m_note.def("getInferedOnsets", &getInferedOnsets);
    m_note.def("modelOutput2Notes", &modelOutput2Notes);
    py::class_<Note>(m_note, "Note")
        .def_readwrite("start", &Note::start_time)
        .def_readwrite("end", &Note::end_time)
        .def_readwrite("pitch", &Note::pitch)
        .def_readwrite("amplitude", &Note::amplitude)
        .def("__repr__",
        [] (const Note &note) {
            return std::to_string(note.start_time) + "\t"
                + std::to_string(note.end_time) + "\t"
                + std::to_string(note.pitch) + "\t"
                + std::to_string(note.amplitude);
        })
        ;
}

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
        .def("transcribeAudio", &amtModel::transcribeAudio)
        .def("getOutput", &amtModel::getOutput)
        .def("getCQ", &amtModel::getCQ)
        ;
}


// bind the utils functions
void bind_utils( py::module &m ) {
    auto m_utils = m.def_submodule("utils");
    m_utils.def("testMatConversion", [] ( py::array_t<float> input ) {
        printPyarray(input);
        VecMatrixf input_tensor = pyarray2mat3D(input);
        printVecMatrixf(input_tensor);
        auto output = mat3D2pyarray(input_tensor);
        printPyarray(output);
        return output;
    });
    m_utils.def("getWindowedAudio", &getWindowedAudio);
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
        .def("computeCQT", &CQ::computeCQT, py::arg("x"), py::arg("batch_norm") = false)
        .def("getKernel", &CQ::getKernel)
        .def("getFilter", &CQ::getFilter)
        .def("harmonicStacking", [] ( CQ &cq, Vectorf &x, bool batch_norm ) {
            VecMatrixf output_tensor = cq.cqtHarmonic(x, batch_norm);
            return mat3D2pyarray(output_tensor);
        }, py::arg("x"), py::arg("batch_norm") = false);
}

PYBIND11_MODULE(BasiCPP_Pitch, m) {
    m.doc() = "BasiCPP_Pitch: A C++ implementation of the pitch detection algorithm";
    bind_amtModel(m);
    bind_CQParams(m);
    bind_CQ(m);
    bind_layer(m);
    bind_cnn(m);
    bind_note(m);
    bind_utils(m);
    bind_nnUtils(m);
}