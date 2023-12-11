#pragma once

#include "typedef.h"
#include "CQT.h"
#include "cnn.h"
#include "note.h"

class amtModel {
    public:

        amtModel();

        ~amtModel() = default;

        // reset the model
        void reset();

        // transcriibe audio
        std::vector<Note> transcribeAudio( const Vectorf& audio );

        // inference API for Eigen IO
        void inferenceFrame( const Vectorf& x );


        // get the CQ object, just for testing
        CQ getCQ() { return _cqt; }

        // get the buffer for Yp, Yn, Yo
        VecVecMatrixf getBuffer() { return {_Yp_buffer, _Yn_buffer, _Yo_buffer};}

        VecMatrixf getOutput();

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

        // Buffer for Yp, Yn, Yo
        VecMatrixf _Yp_buffer;
        VecMatrixf _Yn_buffer;
        VecMatrixf _Yo_buffer;

        int _audio_len;
};



