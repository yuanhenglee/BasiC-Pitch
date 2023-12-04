#include "layer.h"
#include "utils.h"
#include <iostream>
#include <string>

Conv2D::Conv2D(
    int n_filters_in,
    int n_filters_out,
    int n_features_in,
    int kernel_size_time,
    int kernel_size_feature,
    int stride
) :
    _n_filters_in( n_filters_in ),
    _n_filters_out( n_filters_out ),
    _n_features_in( n_features_in ),
    _n_features_out( computeNFeaturesOut(n_features_in, kernel_size_feature, stride) ),
    _kernel_size_time( kernel_size_time ),
    _kernel_size_feature( kernel_size_feature ),
    _stride( stride ),
    Layer( n_features_in * n_filters_in, _n_features_out * n_filters_out ) {

    std::cout << get_name() << " constructor called" << std::endl;

}

Conv2D::Conv2D( int& json_idx, const json& weights ) : 
    Layer( (loadWeights( json_idx, weights ),_n_features_in * _n_filters_in), _n_features_out * _n_filters_out ) {
    // loadWeights( json_idx, weights );
    std::cout << get_name() << " constructor called" << std::endl;
    // Layer( _n_features_in * _n_filters_in, _n_features_out * _n_filters_out );
}

Conv2D::~Conv2D() {
    std::cout << get_name() << " destructor called" << std::endl;
}

std::string Conv2D::get_name() const{
    
    return std::to_string(_n_filters_out) + 
        " Conv2D (" +
        "x" + std::to_string(_kernel_size_time) + 
        "x" + std::to_string(_kernel_size_feature) + 
        ")";
}

Tensor3f Conv2D::forward( const Tensor3f& input ) const{
    std::cout << get_name() << " forward pass" << std::endl;
}

void Conv2D::loadWeights( int& json_idx, const json& w_json ){
    std::cout << get_name() << " loading weights" << std::endl;

    
    _n_filters_in = w_json["num_filters_in"].get<int>();
    _n_filters_out = w_json["num_filters_out"].get<int>();
    _n_features_in = w_json["num_features_in"].get<int>();
    _kernel_size_time = w_json["kernel_size_time"].get<int>();
    _kernel_size_feature = w_json["kernel_size_feature"].get<int>();
    _stride = w_json["strides"].get<int>();
    _n_features_out = computeNFeaturesOut(_n_features_in, _kernel_size_feature, _stride);

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