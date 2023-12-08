#include "layer.h"
#include "utils.h"
#include <iostream>
#include <string>

// Conv2D::Conv2D(
//     int n_filters_in,
//     int n_filters_out,
//     int n_features_in,
//     int kernel_size_time,
//     int kernel_size_feature,
//     int stride
// ) :
//     _n_filters_in( n_filters_in ),
//     _n_filters_out( n_filters_out ),
//     _n_features_in( n_features_in ),
//     _n_features_out( computeNFeaturesOut(n_features_in, kernel_size_feature, stride) ),
//     _kernel_size_time( kernel_size_time ),
//     _kernel_size_feature( kernel_size_feature ),
//     _stride( stride ),
//     Layer() {
//     _input_size = _n_filters_in * _n_features_in;
//     _output_size = _n_filters_out * _n_features_out;
//     std::cout << get_name() << " constructor called" << std::endl;
// }

Conv2D::Conv2D( int& json_idx, const json& weights ) : Layer() {
    loadWeights( json_idx, weights );
}

std::string Conv2D::get_name() const{
    
    return std::to_string(_n_filters_out) + 
        " Conv2D (" +
        std::to_string(_kernel_size_time) + 
        "x" + std::to_string(_kernel_size_feature) + 
        ")";
}

// shape of input: ( n_filters_in, n_features_in, n_frames ) a.k.a. ( n_harmonics, n_bins, n_frames )
// shape of output: ( n_filters_out, n_features_out, n_frames )
VecMatrixf Conv2D::forward( const VecMatrixf& input ) const{
    std::cout << get_name() << " forward pass" << std::endl;
    int n_frames = input[0].cols();
    VecMatrixf output(_n_filters_out, Matrixf::Zero(_n_features_out, n_frames));

    for ( int i = 0 ; i < _n_filters_in ; i++ ) {
        for ( int j = 0 ; j < _n_filters_out ; j++ ) {
            output[j] += conv2d(input[i], _weights[i][j], _stride);
        }
    }

    return output;
}

void Conv2D::loadWeights( int& json_idx, const json& w_json ){
    _n_filters_in = w_json["num_filters_in"].get<int>();
    _n_filters_out = w_json["num_filters_out"].get<int>();
    _n_features_in = w_json["num_features_in"].get<int>();
    _kernel_size_time = w_json["kernel_size_time"].get<int>();
    _kernel_size_feature = w_json["kernel_size_feature"].get<int>();
    _stride = w_json["strides"].get<int>();
    _n_features_out = computeNFeaturesOut(_n_features_in, _kernel_size_feature, _stride);
    _input_size = _n_filters_in * _n_features_in;
    _output_size = _n_filters_out * _n_features_out;

    // shape of the Tensorflow weights json: ( kernel_size_time, kernel_size_feature, n_filters_in, n_filters_out )
    // shape of _weights: ( n_filters_in, n_filters_out, kernel_size_time, kernel_size_feature )
    const json& weights = w_json["weights"];
    auto layer_weights = weights.at(0);
    _weights.resize( _n_filters_in );
    for ( size_t i = 0 ; i < _n_filters_in ; i++ ) {
        _weights[i].resize( _n_filters_out );
        for ( size_t j = 0 ; j < _n_filters_out ; j++ ) {
            _weights[i][j] = Matrixf::Zero( _kernel_size_time, _kernel_size_feature );
        }
    }

    for ( size_t i = 0 ; i < _kernel_size_time ; i++ ) {
        auto l1 = layer_weights.at(i);
        for ( size_t j = 0 ; j < _kernel_size_feature ; j++ ) {
            auto l2 = l1.at(j);
            for ( size_t k = 0 ; k < _n_filters_in ; k++ ) {
                auto l3 = l2.at(k);
                for ( size_t l = 0 ; l < _n_filters_out ; l++ ) {
                    _weights[k][l](i, j) = l3.at(l).get<float>();
                }
            }
        }
    }

    // bias should be of shape ( n_filters_out )
    auto layer_bias = weights.at(1);
    _bias.resize( _n_filters_out );
    for ( size_t i = 0 ; i < _n_filters_out ; i++ ) {
        _bias(i) = layer_bias.at(i).get<float>();
    }

    if(!w_json.contains("activation")) {
        json_idx++;
    }
    else {
        const std::string activationType = w_json["activation"].get<std::string>();
        if(activationType.empty())
            json_idx++;
    }
}

std::string ReLU::get_name() const{
    return "ReLU";
}

VecMatrixf ReLU::forward( const VecMatrixf& input ) const{
    VecMatrixf output(input);
    std::vector<int> shape = {input.size(), input[0].rows(), input[0].cols()};
    for ( int i = 0 ; i < shape[0] ; i++ )
        for ( int j = 0 ; j < shape[1] ; j++ )
            for ( int k = 0 ; k < shape[2] ; k++ )
                if ( output[i](j, k) < 0 )
                    output[i](j, k) = 0;
    return output;
}

std::string Sigmoid::get_name() const{
    return "Sigmoide";
}

VecMatrixf Sigmoid::forward( const VecMatrixf& input ) const{
    VecMatrixf output(input);
    std::vector<int> shape = {input.size(), input[0].rows(), input[0].cols()};
    for ( int i = 0 ; i < shape[0] ; i++ )
        for ( int j = 0 ; j < shape[1] ; j++ )
            for ( int k = 0 ; k < shape[2] ; k++ ) {
                if ( input[i](j, k) > 0 )
                    output[i](j, k) = 1.0f / (1.0f + std::exp(-input[i](j, k)));
                else {
                    float exp_x = std::exp(input[i](j, k));
                    output[i](j, k) = exp_x / (1.0f + exp_x);
                }
            }
    return output;
}