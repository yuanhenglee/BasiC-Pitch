#pragma once

#include "typedef.h"
#include "CQT.h"
#include "cnn.h"

class amtModel {
    public:

        amtModel();

        ~amtModel() = default;

        // transcriibe audio
        Matrixf transcribeAudio( const Vectorf& audio );

        // inference API for Eigen IO
        void inference( const Vectorf& x );

        // get the CQ object, just for testing
        CQ getCQ() { return _cqt; }

        Matrixf getYo() { return _Yo; }

        Matrixf getYp() { return _Yp; }

        Matrixf getYn() { return _Yn; }


    private:

        // CQ for generating features
        CQ _cqt;


        // CNN for onset detection
        CNN _onset_input_cnn;
        CNN _onset_output_cnn;

        // CNN for note detection
        CNN _note_cnn;

        // CNN for contour detection
        CNN _contour_cnn;

        // Buffer for storing the concat of onset and note output
        Matrixf _Yo;

        // Buffer for storing the contour output
        Matrixf _Yp;

        // Buffer for storing the note output
        Matrixf _Yn;

};



