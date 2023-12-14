#include "cnn.h"
#include "constant.h"
#include "loader.h"
#include <iostream>

CNN::CNN( const std::string model_name ) : _model_name( model_name ) {
    // std::cout << "CNN " + model_name + " constructor called" << std::endl;
    loadCNNModel( _layers, model_name );
    // std::cout << get_name() << std::endl;
}

CNN::~CNN() {
    for ( size_t i = 0 ; i < _layers.size() ; i++ ) {
        delete _layers[i];
    }
}

VecMatrixf CNN::forward( const VecMatrixf& input ) const {
    // std::cout << _model_name + " forward pass" << std::endl;
    VecMatrixf output = input;
    for ( size_t i = 0 ; i < _layers.size() ; i++ ) {
        output = _layers[i]->forward( output );
    }
    // std::cout << "output size = " << output.size() << std::endl;
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

std::vector<Layer*> CNN::get_layers() const {
    return _layers;
}