#pragma once

#include "loader.h"
#include "json.hpp"
#include "cnpy.h"
#include <fstream>

using json = nlohmann::json;

void loadDefaultKernel(Matrixcf &kernel) {
    // load the precomputed kernel
    cnpy::NpyArray arr = cnpy::npy_load("model/kernel.npy");
    const size_t& n_bins = arr.shape[0];
    const size_t& kernel_length = arr.shape[1];
    std::complex<float>* data = const_cast<std::complex<float>*>(arr.data<std::complex<float>>());
    kernel = Eigen::Map<Matrixcf>(data, n_bins, kernel_length);
}

void loadDefaultLowPassFilter( Vectorf &filter_kernel) {
    // load the precomputed filter
    cnpy::NpyArray arr = cnpy::npy_load("model/lowpass_filter.npy");
    const size_t& kernel_length = arr.shape[0];
    float* data = const_cast<float*>(arr.data<float>());
    filter_kernel = Eigen::Map<Vectorf>(data, kernel_length);
}

inline const std::string getModelPath( std::string model_name ) {
    if (model_name == "Contour")
        return "model/cnn_contour_model.json";
    else if (model_name == "Onset Input")
        return "model/cnn_onset_1_model.json";
    else if (model_name == "Onset Output")
        return "model/cnn_onset_2_model.json";
    else if (model_name == "Note")
        return "model/cnn_note_model.json";
    else
        return "Unknown model";
}

void loadCNNModel( std::vector<Layer*>& layers, std::string model_name ) {
    // load the model
    std::ifstream f(getModelPath(model_name));
    json w_json = json::parse(f);

    int json_idx = 0;
    // load the weights
    auto json_layers = w_json["layers"];
    while( json_idx < json_layers.size() ) {
        auto layer_json = json_layers.at(json_idx);
        std::string layer_type = layer_json["type"].get<std::string>();

        if ( layer_type == "conv2d" ) {
            layers.emplace_back(new Conv2D(json_idx, layer_json));
            if ( layer_json.contains("activation") ) {
                std::string activation = layer_json["activation"].get<std::string>();
                if ( activation == "relu" ) {
                    layers.emplace_back(new ReLU());
                    json_idx++;
                }
                else if ( activation == "sigmoid" ) {
                    layers.emplace_back(new Sigmoid());
                    json_idx++;
                }
                else {
                    std::cout << "Unknown activation function: " << activation << std::endl;
                }
            }
        }
        else if ( layer_type == "relu") {
            layers.emplace_back(new ReLU());
            json_idx++;
        }
        else if ( layer_type == "sigmoid") {
            layers.emplace_back(new Sigmoid());
            json_idx++;
        }
        else if ( layer_type == "batchnorm2d" ) {
            layers.emplace_back(new BatchNorm(json_idx, layer_json));
        }
        else {
            std::cout << "Unknown layer type: " << layer_type << std::endl;
        }
    }

    f.close();
}

Vectorf getExampleAudio() {
    // load the example audio
    cnpy::NpyArray arr = cnpy::npy_load("data/Undertale-Megalovania.npy");
    const size_t& audio_length = arr.shape[0];
    float* data = const_cast<float*>(arr.data<float>());
    Vectorf example_audio = Eigen::Map<Vectorf>(data, audio_length);
    return example_audio;
}