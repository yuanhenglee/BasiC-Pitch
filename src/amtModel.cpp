#include "amtModel.h"
#include <iostream>

amtModel::amtModel(): 
    _cqt(),
    _onset_input_cnn("Onset Input"),
    _onset_output_cnn("Onset Output"),
    _note_cnn("Note"),
    _contour_cnn("Contour") {}

void amtModel::inference( const Vectorf& x ) {
    // compute harmonic stacking, shape : (n_harmonics, n_frames, n_bins)
    VecMatrixf cqt = _cqt.cqtHarmonic(x);


    VecMatrixf contour_out = _contour_cnn.forward(cqt);
    _Yp = contour_out[0];

    VecMatrixf note_out = _note_cnn.forward(contour_out);
    _Yn = note_out[0];

    VecMatrixf concat_buf = {_Yn};
    VecMatrixf onset_out = _onset_input_cnn.forward(cqt);
    concat_buf.insert(concat_buf.end(), onset_out.begin(), onset_out.end());

    VecMatrixf concat_out = _onset_output_cnn.forward(concat_buf);
    _Yo = concat_out[0];
}