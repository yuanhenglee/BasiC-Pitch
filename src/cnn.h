#pragma once

#include "typedef.h"
#include "layer.h"
#include <vector>
#include <string>


class CNN {
    public:

        CNN( const std::string model_name );

        ~CNN();
    
        // inference API for Eigen IO
        VecMatrixf forward( const VecMatrixf& input ) const;

        std::string get_name() const;

        std::vector<Layer*> get_layers() const;    

    // private:
    protected:

        std::vector<Layer*> _layers;
    
        std::string _model_name;
};