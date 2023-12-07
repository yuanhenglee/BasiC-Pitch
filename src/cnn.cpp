#include "cnn.h"
#include "constant.h"
#include "loader.h"
#include <iostream>

CNN::CNN( const std::string model_name ) : _model_name( model_name ) {
    std::cout << "CNN " + model_name + " constructor called" << std::endl;
    loadCNNModel( _layers, model_name );
    std::cout << get_name() << std::endl;
}

CNN::~CNN() {
    for ( size_t i = 0 ; i < _layers.size() ; i++ ) {
        delete _layers[i];
    }
    // _layers.clear();
}

Tensor3f CNN::forward( const Tensor3f& input ) const {
    std::cout << "CNN forward pass" << std::endl;
    Tensor3f output = input;
    for ( size_t i = 0 ; i < _layers.size() ; i++ ) {
        output = _layers[i]->forward( output );
    }
    return output;
}

std::string CNN::get_name() const {
    std::string name = _model_name + " <\n";
    for ( size_t i = 0 ; i < _layers.size() ; i++ ) {
        name += "\t" + _layers[i]->get_name() + "\n";
    }
    name += ">";
    return name;
}