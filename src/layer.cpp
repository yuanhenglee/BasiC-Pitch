#include "layer.h"
#include "nnUtils.h"
#include <iostream>
#include <string>

Conv2D::Conv2D( int& json_idx, const json& weights ) : Layer(LayerType::CONV2D) {
    loadWeights( json_idx, weights );
}

std::string Conv2D::get_name() const{
    
    return std::to_string(_n_filters_out) + 
        " Conv2D (" +
        std::to_string(_kernel_size_time) + 
        "x" + std::to_string(_kernel_size_feature) + 
        ")";
}

VecMatrixf Conv2D::forward( const VecMatrixf& input ) const {

    // return forward_naive(input);
    return forward_im2col(input);
}

// naive implementation of 2D convolution
VecMatrixf Conv2D::forward_naive( const VecMatrixf& input ) const{
    // std::cout << "\t" << get_name() << " forward pass" << std::endl;
    int n_frames_in = input[0].rows();
    int n_frames_out = n_frames_in;
    VecMatrixf output(_n_filters_out, Matrixf::Zero(n_frames_out, _n_features_out));

    for ( int i = 0 ; i < _n_filters_in ; i++ ) {
        for ( int j = 0 ; j < _n_filters_out ; j++ ) {
            output[j] += conv2d(input[i], _weights[i][j], _stride);
        }
    }

    for ( int i = 0 ; i < _n_filters_out ; i++ ) {
        output[i].array() += _bias[i];
    }

    return output;
}

// im2col + gemm implementation of 2D convolution
VecMatrixf Conv2D::forward_im2col( const VecMatrixf& input ) const {
    int n_frames_out = input[0].rows();
    Matrixf input2cols = im2col(input, n_frames_out, _n_features_out, _kernel_size_time, _kernel_size_feature, _stride);

    // gemm
    Matrixf output2cols = _weights_2cols * input2cols;

    // add bias
    for ( int i = 0 ; i < _n_filters_out ; i++ ) {
        output2cols.row(i).array() += _bias[i];
    }

    VecMatrixf output = col2im(output2cols, n_frames_out, _n_features_out);
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

    // shape of the Tensorflow weights json: ( kernel_size_time, kernel_size_feature, n_filters_in, n_filters_out )
    // shape of _weights: ( n_filters_in, n_filters_out, kernel_size_time, kernel_size_feature )
    // shape of _weights_2cols: ( n_filters_out, n_filters_in * kernel_size_time * kernel_size_feature )
    const json& weights = w_json["weights"];
    auto layer_weights = weights.at(0);
    _weights.resize( _n_filters_in );
    _weights_2cols = Matrixf::Zero( _n_filters_out, _n_filters_in * _kernel_size_time * _kernel_size_feature );
    
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
                    float w = l3.at(l).get<float>();
                    _weights[k][l](i, j) = w;
                    _weights_2cols(l, k * _kernel_size_time * _kernel_size_feature + i * _kernel_size_feature + j) = w;
                }
            }
        }
    }

    // bias should be of shape ( n_filters_out )
    auto layer_bias = weights.at(1);
    _bias = layer_bias.get<std::vector<float>>();

    if ( _bias.size() != _n_filters_out ) {
        std::cout << "Error: bias size mismatch" << std::endl;
        exit(1);
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

VecVecMatrixf Conv2D::getWeights() const{
    return _weights;
}

ReLU::ReLU() : Layer(LayerType::RELU) {}

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

Sigmoid::Sigmoid() : Layer(LayerType::SIGMOID) {}

std::string Sigmoid::get_name() const{
    return "Sigmoid";
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

BatchNorm::BatchNorm( int& json_idx, const json& weights ) : Layer(LayerType::BATCHNORM) {
    loadWeights( json_idx, weights );
}

std::string BatchNorm::get_name() const{
    return "BatchNorm";
}

VecMatrixf BatchNorm::forward( const VecMatrixf& input ) const{
    VecMatrixf output(input);
    for ( int i = 0 ; i < input.size() ; i++ ) {
        output[i] = (input[i].array() - _mean[i]) * _multiplier[i] + _beta[i];
    }
    return output;
}

void BatchNorm::loadWeights( int& json_idx, const json& w_json ){
    
    const json& weights = w_json["weights"];
    _gamma = weights.at(0).get<std::vector<float>>();
    _beta = weights.at(1).get<std::vector<float>>();
    _mean = weights.at(2).get<std::vector<float>>();
    _variance = weights.at(3).get<std::vector<float>>();
    
    // calculate multiplier
    _multiplier.resize(_gamma.size());
    for ( int i = 0 ; i < _gamma.size() ; i++ ) {
        _multiplier[i] = _gamma[i] / std::sqrt(_variance[i] + 0.001f);
    }

    json_idx++;
}