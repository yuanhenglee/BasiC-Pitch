#pragma once

#include "typedef.h"
#include <string>

class Layer {
    public:

        Layer( int input_size, int output_size ) : _input_size( input_size ), _output_size( output_size ){}

        virtual ~Layer() = default;

        virtual std::string get_name() const = 0;

        virtual void forward( const float* input, float* output ) const = 0;

    private:
        int _input_size;
        int _output_size;
};

class Conv2D : public Layer {
    public:

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

        void forward( const float* input, float* output ) const override;

    private:
        int _n_filters_in;
        int _n_filters_out;
        int _n_features_in;
        int _kernel_size_time;
        int _kernel_size_feature;
        int _stride;

};