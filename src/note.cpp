#include "note.h"

std::vector<Note> modelOutput2Notes( const Matrixf& Yp, const Matrixf& Yn, const Matrixf& Yo ) {
    std::vector<Note> notes;
    
    Matrixf infered_Yo = getInferedOnsets( Yn, Yo );

    return notes;
}

Matrixf getInferedOnsets( const Matrixf& Yo, const Matrixf& Yn ) {

    // int n_diff = 2;
    Matrixf shifted_Yn = Matrixf::Zero( Yn.rows(), Yn.cols() );
    shifted_Yn.block( 2, 0, Yn.rows() - 2, Yn.cols() ) = Yn.block( 0, 0, Yn.rows() - 2, Yn.cols() );
    Matrixf diff2 = Yn - shifted_Yn;
    shifted_Yn.block( 1, 0, Yn.rows() - 1, Yn.cols() ) = Yn.block( 0, 0, Yn.rows() - 1, Yn.cols() );
    Matrixf diff = Yn - shifted_Yn;
    diff = diff.cwiseMin( diff2 ).cwiseMax(0);
    diff.block( 0, 0, 2, diff.cols() ) *= 0;
    return  Yo.cwiseMax( diff * Yo.maxCoeff() / diff.maxCoeff() );

}