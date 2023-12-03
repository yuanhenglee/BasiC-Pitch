#include "cnn.h"
#include <iostream>

CNN::CNN( int input_size, int output_size) : _input_size(input_size), _output_size(output_size) {
    // TODO : load weights
    std::cout << "CNN initialized" << std::endl;

    Layer* layer1 = new Conv2D( 1, 1, 1, 5, 5, 1 );
    Layer* layer2 = new Conv2D( 1, 1, 1, 3, 3, 1 );

    _layers.push_back( layer1 );
    _layers.push_back( layer2 );
}

CNN::~CNN() {
    // for ( size_t i = 0 ; i < _layers.size() ; i++ ) {
    //     delete _layers[i];
    // }
    // _layers.clear();
    ;
}

void CNN::forward( const float* input, float* output ) {
    std::cout << "CNN forward pass" << std::endl;

    for ( size_t i = 0 ; i < _layers.size() ; i++ ) {
        // TODO: allocate memory for output 
        _layers[i]->forward( input, output );
    }
}

ContourCNN::ContourCNN() : CNN( 1, 1 ) {
    std::cout << "ContourCNN initialized" << std::endl;
}

std::string ContourCNN::get_name() const {
    std::string name = "ContourCNN <\n";
    for ( size_t i = 0 ; i < _layers.size() ; i++ ) {
        name += _layers[i]->get_name() + "\n";
    }
    name += ">";
    return name;
}