#pragma once

#include "typedef.h"
#include <vector>

struct Note {
    // time in seconds
    float start_time;
    float end_time;
    // frame number
    int start_frame;
    int end_frame;
    int pitch; // MIDI pitch
    float amplitude;
    std::vector<int> bends; // units of 1/3 semitone
};

std::vector<Note> modelOutput2Notes( const Matrixf& Yp, const Matrixf& Yn, const Matrixf& Yo, const bool melodia_trick = true );

Matrixf getInferedOnsets( const Matrixf& Yo, const Matrixf& Yn );