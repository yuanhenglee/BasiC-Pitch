#pragma once

#include "typedef.h"
#include "json.hpp"
#include <string>

using json = nlohmann::json;

class Layer {
    public:

        Layer() = default;

        virtual ~Layer() = default;

        virtual std::string get_name() const = 0;

        virtual Tensor3f forward( const Tensor3f& input ) const = 0;

        virtual void loadWeights( int& json_idx, const json& weights ) = 0;

    protected:

        int _input_size;
        int _output_size;
};

class Conv2D : public Layer {
    public:

        Conv2D( int& json_idx, const json& weights );
        Conv2D(
            int n_filters_in,
            int n_filters_out,
            int n_features_in,
            int kernel_size_time,
            int kernel_size_feature,
            int stride
        );

        ~Conv2D();

        std::string get_name() const override;

        Tensor3f forward( const Tensor3f& input ) const override;

        void loadWeights( int& json_idx, const json& weights ) override;

    private:

        int _n_filters_in;
        int _n_filters_out;
        int _n_features_in;
        int _n_features_out;
        int _kernel_size_time;
        int _kernel_size_feature;
        int _stride;

        Tensor4f _weights;
        Vectorf _bias;

};