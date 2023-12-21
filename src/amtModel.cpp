#include "amtModel.h"
#include "utils.h"
#include "constant.h"
#include <iostream>

amtModel::amtModel(): 
    _cqt(),
    _onset_input_cnn("Onset Input"),
    _onset_output_cnn("Onset Output"),
    _note_cnn("Note"),
    _contour_cnn("Contour") {}

void amtModel::reset() {
    _audio_len = 0;
    _Yp_buffer.clear();
    _Yn_buffer.clear();
    _Yo_buffer.clear();
}

std::vector<Note> amtModel::transcribeAudio( const Vectorf& audio ) {

    // reset the model
    reset();

    _audio_len = audio.size();
    auto audio_windowed = getWindowedAudio(audio);
    for ( Vectorf& x : audio_windowed ) {
        inferenceFrame(x);
    }

    int n_frames_in = _Yn_buffer[0].cols() / ANNOTATIONS_N_SEMITONES;
    // concat 3 buffers
    Matrixf Yp = concatMatrices(_Yp_buffer, _audio_len, n_frames_in);
    Matrixf Yn = concatMatrices(_Yn_buffer, _audio_len, n_frames_in);
    Matrixf Yo = concatMatrices(_Yo_buffer, _audio_len, n_frames_in);

    // convert to midi note events
    return modelOutput2Notes(Yp, Yn, Yo, true);
}

// input shape : (N_AUDIO_SAMPLES, N_BIN_CONTORU )
void amtModel::inferenceFrame( const Vectorf& x ) {

    // compute harmonic stacking, shape : (n_harmonics, n_frames, n_bins)
    Matrixf cqt = _cqt.computeCQT(x, true);

    Matrixf contour_out = _contour_cnn.forward(cqt);
    _Yp_buffer.push_back(contour_out); // Yp

    Matrixf note_out = _note_cnn.forward(contour_out);
    _Yn_buffer.push_back(note_out); // Yn

    Matrixf onset_out = _onset_input_cnn.forward(cqt);
    Matrixf concat_buf(note_out.rows() + onset_out.rows(), note_out.cols());
    concat_buf << note_out, onset_out;

    Matrixf concat_out = _onset_output_cnn.forward(concat_buf);
    _Yo_buffer.push_back(concat_out); // Yo

}

VecMatrixf amtModel::getOutput() {
    return {_Yp, _Yn, _Yo};
}