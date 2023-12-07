#include "amtModel.h"

amtModel::amtModel(): 
    _cqt(),
    _onset_input_cnn("Onset Input"),
    _onset_output_cnn("Onset Output"),
    _note_cnn("Note"),
    _contour_cnn("Contour") {}

void amtModel::inference(
    const Vectorf& x,
    Vectorf& onset_output, // Y_o in the paper
    Vectorf& note_output, // Y_n in the paper
    Vectorf& contour_output // Y_p in the paper, 3 bins per semitone
) {
    // compute harmonic stacking, shape : (n_harmonics, n_bins, n_frames)
    VecMatrixf cqt = _cqt.cqtHarmonic(x);


    // TODO: compute onset_output, note_output, contour_output

}

CQ amtModel::getCQ() {
    return _cqt;
}