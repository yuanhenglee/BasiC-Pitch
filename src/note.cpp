#include "note.h"
#include "constant.h"
#include <iostream>

inline float modelFrames2Time( int frame ) {
    return static_cast<float>(frame * FFT_HOP) / SAMPLE_RATE - WINDOW_OFFSET * ( frame / ANNOT_N_FRAMES );
}

std::vector<Note> modelOutput2Notes( const Matrixf& Yp, const Matrixf& Yn, const Matrixf& Yo ) {

    int n_frames = Yn.rows(), n_pitches = Yn.cols();

    std::vector<Note> notes;
    
    Matrixf infered_Yo = getInferedOnsets( Yn, Yo );
    Matrixf remaining_energy(Yn);

    // loop over onsets, go backwards in time
    // skip the last frame as line 399 in basic_pitch/note_creation.py
    for ( int start_idx = n_frames -2 ; start_idx >= 0 ; --start_idx ) {
        for ( int note_idx = n_pitches-1 ; note_idx >= 0 ; --note_idx ){

            float onset = infered_Yo(start_idx, note_idx);
            // skip if onset is below threshold
            if ( onset < ONSET_THRESHOLD )
                continue;
            
            // check if onset is peak
            float prev_onset = start_idx > 0 ? infered_Yo(start_idx-1, note_idx) : onset;
            float next_onset = start_idx < n_frames-1 ? infered_Yo(start_idx+1, note_idx) : onset;
            if ( onset < prev_onset || onset < next_onset )
                continue;

            // find time index at this frequency band where the frames drop below an energy threshold
            int i  = start_idx + 1, k = 0;
            while( i < n_frames - 1 && k < ENERGY_THRESHOLD ) {
                if ( remaining_energy(i, note_idx) < FRAME_THRESHOLD )
                    k++;
                else
                    k = 0;
                ++i;
            }
            i -= k; // go back to frame above threshold

            // if the note is too short, skip it
            if ( i - start_idx < MIN_NOTE_LENGTH )
                continue;

            remaining_energy.block(start_idx, note_idx, i - start_idx, 1) *= 0;
            if ( note_idx < n_pitches - 1 )
                remaining_energy.block(start_idx, note_idx + 1, i - start_idx, 1) *= 0;
            if ( note_idx > 0 )
                remaining_energy.block(start_idx, note_idx - 1, i - start_idx, 1) *= 0;

            // add the note
            float amplitude = Yn.block(start_idx, note_idx, i - start_idx, 1).mean();
            notes.emplace_back( Note{
                modelFrames2Time(start_idx),
                modelFrames2Time(i),
                start_idx,
                i,
                note_idx + MIDI_OFFSET,
                amplitude,
                std::vector<int>()
            } );
        }
    }

    return notes;
}

Matrixf getInferedOnsets( const Matrixf& Yo, const Matrixf& Yn ) {

    Matrixf shifted_Yn = Matrixf::Zero( Yn.rows(), Yn.cols() );
    shifted_Yn.block( 2, 0, Yn.rows() - 2, Yn.cols() ) = Yn.block( 0, 0, Yn.rows() - 2, Yn.cols() );
    Matrixf diff2 = Yn - shifted_Yn;
    shifted_Yn.block( 1, 0, Yn.rows() - 1, Yn.cols() ) = Yn.block( 0, 0, Yn.rows() - 1, Yn.cols() );
    Matrixf diff = Yn - shifted_Yn;
    diff = diff.cwiseMin( diff2 ).cwiseMax(0);
    diff.block( 0, 0, 2, diff.cols() ) *= 0;
    return  Yo.cwiseMax( diff * Yo.maxCoeff() / diff.maxCoeff() );

}