#pragma once

#include "typedef.h"
#include "json.hpp"
#include <string>
#include <vector>

using json = nlohmann::json;

enum LayerType {
    NONE,
    CONV2D,
    RELU,
    SIGMOID,
    BATCHNORM
};

class Layer {
    public:

        Layer(LayerType type) : type(type) {}

        ~Layer() = default;

        virtual std::string get_name() const = 0;

        // virtual VecMatrixf forward( const VecMatrixf& input ) const = 0;
        
        // forward for (C, HW) input
        virtual Matrixf forward( const Matrixf& input ) const = 0;

        LayerType type;
};

class Conv2D : public Layer {
    public:

        Conv2D( int& json_idx, const json& weights );

        std::string get_name() const override;

        // VecMatrixf forward( const VecMatrixf& input ) const override;

        Matrixf forward( const Matrixf& input ) const override;

        void loadWeights( int& json_idx, const json& weights );

        VecVecMatrixf getWeights() const;


    private:

        // different forward implementation
        VecMatrixf forward_naive( const VecMatrixf& input ) const;

        VecMatrixf forward_im2col( const VecMatrixf& input ) const;

        int _n_filters_in;
        int _n_filters_out;
        int _n_features_in;
        int _n_features_out;
        int _kernel_size_time;
        int _kernel_size_feature;
        int _stride;

        VecVecMatrixf _weights;
        // im2col version of weights, shape: ( n_filters_out, n_filters_in * kernel_size_time * kernel_size_feature)
        Matrixf _weights_2cols;
        std::vector<float> _bias;
        ColVectorf _bias_vec;

};

class ReLU : public Layer {
    public:

        ReLU();

        std::string get_name() const override;

        // VecMatrixf forward( const VecMatrixf& input ) const override;

        Matrixf forward( const Matrixf& input ) const override;

};

class Sigmoid : public Layer {
    public:

        Sigmoid();

        std::string get_name() const override;

        // VecMatrixf forward( const VecMatrixf& input ) const override;

        Matrixf forward( const Matrixf& input ) const override;

};

class BatchNorm : public Layer {
    public:

        BatchNorm( int& json_idx, const json& weights );

        std::string get_name() const override;

        // VecMatrixf forward( const VecMatrixf& input ) const override;

        Matrixf forward( const Matrixf& input ) const override;

        void loadWeights( int& json_idx, const json& weights );

    private:
        int _n_filters_in;
        std::vector<float> _mean;
        std::vector<float> _variance;
        std::vector<float> _gamma;
        std::vector<float> _beta;
        std::vector<float> _multiplier;

        ColArrayf _mean_vec;
        ColArrayf _variance_vec;
        ColArrayf _gamma_vec;
        ColArrayf _beta_vec;
        ColArrayf _multiplier_vec;

};