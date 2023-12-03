#include "amtModel.h"

amtModel::amtModel() = default;

void amtModel::inference(
    const Vectorf& x,
    Vectorf& onset_output, // Y_o in the paper
    Vectorf& note_output, // Y_n in the paper
    Vectorf& contour_output // Y_p in the paper, 3 bins per semitone
) {
    // compute harmonic stacking
    Tensor3f cqt = _cqt.cqtHarmonic(x);

    // TODO: compute onset_output, note_output, contour_output

}

CQ amtModel::getCQ() {
    return _cqt;
}