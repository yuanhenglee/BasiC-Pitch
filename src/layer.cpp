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

Tensor3f Conv2D::forward( const Tensor3f& input ) const{
    std::cout << get_name() << " forward pass" << std::endl;
    return input;
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
    const json& weights = w_json["weights"];
    auto layer_weights = weights.at(0);
    _weights = Tensor4f( _kernel_size_time, _kernel_size_feature, _n_filters_in, _n_filters_out);
    for ( size_t i = 0 ; i < _kernel_size_time ; i++ ) {
        auto l1 = layer_weights.at(i);
        for ( size_t j = 0 ; j < _kernel_size_feature ; j++ ) {
            auto l2 = l1.at(j);
            for ( size_t k = 0 ; k < _n_filters_in ; k++ ) {
                auto l3 = l2.at(k);
                for ( size_t l = 0 ; l < _n_filters_out ; l++ ) {
                    _weights(i, j, k, l) = l3.at(l).get<float>();
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

Tensor3f ReLU::forward( const Tensor3f& input ) const{
    Tensor3f output = input.cwiseMax(0.0f);
    return output;
}

std::string Sigmoid::get_name() const{
    return "Sigmoide";
}

Tensor3f Sigmoid::forward( const Tensor3f& input ) const{
   Tensor3f output;
   for ( int i = 0 ; i < input.dimension(0) ; i++ ) {
       for ( int j = 0 ; j < input.dimension(1) ; j++ ) {
           for ( int k = 0 ; k < input.dimension(2) ; k++ ) {
               if ( input(i, j, k) > 0 ) {
                   output(i, j, k) = 1.0f / (1.0f + std::exp(-input(i, j, k)));
               }
               else {
                   float exp_x = std::exp(input(i, j, k));
                   output(i, j, k) = exp_x / (1.0f + exp_x);
               }
           }
       }
   }
    
    return output;
}