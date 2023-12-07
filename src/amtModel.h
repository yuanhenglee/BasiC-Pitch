#pragma once

#include "typedef.h"
#include "CQT.h"
#include "cnn.h"

class amtModel {
    public:

        amtModel();

        ~amtModel() = default;

        // inference API for Eigen IO
        void inference(
            const Vectorf& x,
            Vectorf& onset_output, // Y_o in the paper
            Vectorf& note_output, // Y_n in the paper
            Vectorf& contour_output // Y_p in the paper, 3 bins per semitone
        );

        // get the CQ object, just for testing
        CQ getCQ();

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

};



