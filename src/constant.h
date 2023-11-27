#ifndef CONSTANT_H
#define CONSTANT_H

#include <cmath>

inline constexpr int SAMPLE_RATE = 22050;

inline constexpr int SEMITON_PER_OCTAVE = 12;

inline constexpr int NOTES_BINS_PER_SEMITONE = 1;

inline constexpr int CONTOURS_BINS_PER_SEMITONE = 3;

inline constexpr int ANNOTATIONS_N_SEMITONES = 88;

inline constexpr float MIN_FREQ = 27.5;

inline const int MAX_N_SEMITONES = floor(
    SEMITON_PER_OCTAVE * log2(0.5f * SAMPLE_RATE / MIN_FREQ)
);

inline constexpr int FFT_HOP = 256;

inline constexpr int N_FFT = 8 * FFT_HOP;

inline constexpr int N_BINS_NOTE = ANNOTATIONS_N_SEMITONES * NOTES_BINS_PER_SEMITONE;

inline constexpr int N_BINS_CONTOUR = ANNOTATIONS_N_SEMITONES * CONTOURS_BINS_PER_SEMITONE;

inline constexpr int N_HARMONICS = 8;

inline const int N_SEMITONES = std::min(
    static_cast<int>(
        ceil( SEMITON_PER_OCTAVE * log2(N_HARMONICS)) + ANNOTATIONS_N_SEMITONES
    ),
    MAX_N_SEMITONES
);

#endif // CONSTANT_H