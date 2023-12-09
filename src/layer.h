#pragma once

#include "typedef.h"
#include "json.hpp"
#include <string>
#include <vector>

using json = nlohmann::json;

class Layer {
    public:

        Layer() = default;

        ~Layer() = default;

        virtual std::string get_name() const = 0;

        virtual VecMatrixf forward( const VecMatrixf& input ) const = 0;

    protected:

        int _input_size;
        int _output_size;
};

class Conv2D : public Layer {
    public:

        Conv2D( int& json_idx, const json& weights );
        // Conv2D(
        //     int n_filters_in,
        //     int n_filters_out,
        //     int n_features_in,
        //     int kernel_size_time,
        //     int kernel_size_feature,
        //     int stride
        // );

        std::string get_name() const override;

        VecMatrixf forward( const VecMatrixf& input ) const override;

        void loadWeights( int& json_idx, const json& weights );

        VecVecMatrixf getWeights() const;

    private:

        int _n_filters_in;
        int _n_filters_out;
        int _n_features_in;
        int _n_features_out;
        int _kernel_size_time;
        int _kernel_size_feature;
        int _stride;

        VecVecMatrixf _weights;
        std::vector<float> _bias;

};

class ReLU : public Layer {
    public:

        std::string get_name() const override;

        VecMatrixf forward( const VecMatrixf& input ) const override;

};

class Sigmoid : public Layer {
    public:

        std::string get_name() const override;

        VecMatrixf forward( const VecMatrixf& input ) const override;

};