#include "layer.h"
#include "utils.h"
#include <iostream>

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
    _kernel_size_time( kernel_size_time ),
    _kernel_size_feature( kernel_size_feature ),
    _stride( stride ),
    Layer( n_features_in * n_filters_in, 
        computeNFeaturesOut(n_features_in, kernel_size_feature, stride) * n_filters_out
    )
{

    std::cout << get_name() << " constructor called" << std::endl;

}

Conv2D::~Conv2D()
{
    std::cout << get_name() << " destructor called" << std::endl;
}

std::string Conv2D::get_name() const{
    
    return std::to_string(_n_filters_out) + 
        " Conv2D (" +
        "x" + std::to_string(_kernel_size_time) + 
        "x" + std::to_string(_kernel_size_feature) + 
        ")";
}

void Conv2D::forward( const float* input, float* output ) const{
    std::cout << get_name() << " forward pass" << std::endl;
}