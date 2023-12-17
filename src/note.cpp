#include "note.h"
#include "constant.h"
#include <iostream>
#include <vector>
#include <tuple>

inline float modelFrames2Time( int frame ) {
    return (frame * FFT_HOP) / static_cast<double>(SAMPLE_RATE) - WINDOW_OFFSET * std::floor( frame / ANNOT_N_FRAMES );
}

std::vector<Note> modelOutput2Notes( const Matrixf& Yp, const Matrixf& Yn, const Matrixf& Yo, const bool melodia_trick ) {

    int n_frames = Yn.rows(), n_pitches = Yn.cols();

    std::vector<Note> notes;
    
    // constrainFreq( Yo, Yn, MIN_FREQ, MAX_FREQ );
    Matrixf infered_Yo = getInferedOnsets( Yo, Yn );
    Matrixf remaining_energy(Yn);
    std::vector<std::tuple<float*, int, int>> remaining_energy_idices;
    if (melodia_trick) remaining_energy_idices.reserve(n_frames * n_pitches);

    // DEBUG
    int peak_cnt = 0, valid_peak_cnt = 0, long_note_cnt = 0;

    // loop over onsets, go backwards in time
    // skip the last frame as line 399 in basic_pitch/note_creation.py
    // skip start_idx == 0 since they are not consider relmax in scipy.signal.argrelmax
    for ( int start_idx = n_frames -2 ; start_idx > 0 ; --start_idx ) {
        for ( int note_idx = n_pitches-1 ; note_idx >= 0 ; --note_idx ){

            if (melodia_trick)
                remaining_energy_idices.emplace_back( &remaining_energy(start_idx, note_idx), start_idx, note_idx );

            float onset = infered_Yo(start_idx, note_idx);

            // check if onset is peak
            float prev_onset = infered_Yo(start_idx-1, note_idx);
            float next_onset = infered_Yo(start_idx+1, note_idx);
            if ( onset < prev_onset || onset < next_onset )
                continue;
            ++peak_cnt;

            // skip if onset is below threshold
            if ( onset < ONSET_THRESHOLD )
                continue;
            ++valid_peak_cnt;

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
            if ( i - start_idx <= MIN_NOTE_LENGTH )
                continue;
            ++long_note_cnt;


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

    std::cout<<"peak_cnt: "<<peak_cnt<<std::endl;
    std::cout<<"valid_peak_cnt: "<<valid_peak_cnt<<std::endl;
    std::cout<<"long_note_cnt: "<<long_note_cnt<<std::endl;
    std::cout<<"len(notes): "<<notes.size()<<std::endl;

    if (melodia_trick) {
        std::sort( remaining_energy_idices.begin(), remaining_energy_idices.end(),
            [] ( const std::tuple<float*, int, int> &a, const std::tuple<float*, int, int> &b ) {
                return *std::get<0>(a) > *std::get<0>(b);
            }
        );
        for ( auto& remaining_energy_tuple : remaining_energy_idices ) {

            float* max_energy_ptr = std::get<0>(remaining_energy_tuple);
            int i_mid = std::get<1>(remaining_energy_tuple);
            int freq_idx = std::get<2>(remaining_energy_tuple);

            // skip if the energy was set to 0 before
            if ( *max_energy_ptr == 0 )
                continue;

            // break if the energy is below threshold
            if ( *max_energy_ptr <= FRAME_THRESHOLD )
                break;

            remaining_energy(i_mid, freq_idx) = 0;

            // forward pass
            int i, k;
            for ( i = i_mid + 1, k = 0 ; i < n_frames - 1 && k < ENERGY_THRESHOLD ; ++i ) {
                if ( remaining_energy(i, freq_idx) < FRAME_THRESHOLD )
                    k++;
                else
                    k = 0;
                
                remaining_energy(i, freq_idx) = 0;
                if ( freq_idx < n_pitches - 1 )
                    remaining_energy(i, freq_idx + 1) = 0;
                if ( freq_idx > 0 )
                    remaining_energy(i, freq_idx - 1) = 0;
            }
            int i_end = i - 1 - k; // go back to frame above threshold

            // backward pass
            for ( i = i_mid - 1, k = 0 ; i > 0 && k < ENERGY_THRESHOLD ; --i ) {
                if ( remaining_energy(i, freq_idx) < FRAME_THRESHOLD )
                    k++;
                else
                    k = 0;
                
                remaining_energy(i, freq_idx) = 0;
                if ( freq_idx < n_pitches - 1 )
                    remaining_energy(i, freq_idx + 1) = 0;
                if ( freq_idx > 0 )
                    remaining_energy(i, freq_idx - 1) = 0;
            }
            int i_start = i + 1 + k; // go back to frame above threshold

            if ( i_end - i_start <= MIN_NOTE_LENGTH )
                continue; // skip if the note is too short

            float amplitude = Yn.block(i_start, freq_idx, i_end - i_start, 1).mean();

            notes.emplace_back( Note{
                modelFrames2Time(i_start),
                modelFrames2Time(i_end),
                i_start,
                i_end,
                freq_idx + MIDI_OFFSET,
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

inline float hz2midi( float hz ) {
    return 69 + 12 * log2( hz / 440 );
}

void constrainFreq( Matrixf &Yo, Matrixf &Yn, const float min_freq, const float max_freq ) {
    int min_freq_idx = round( hz2midi(min_freq) - MIDI_OFFSET );
    int max_freq_idx = round( hz2midi(max_freq) - MIDI_OFFSET );
    Yo.block( 0, 0, Yo.rows(), min_freq_idx ) *= 0;
    Yo.block( 0, max_freq_idx, Yo.rows(), Yo.cols() - max_freq_idx ) *= 0;
    Yn.block( 0, 0, Yn.rows(), min_freq_idx ) *= 0;
    Yn.block( 0, max_freq_idx, Yn.rows(), Yn.cols() - max_freq_idx ) *= 0;
}