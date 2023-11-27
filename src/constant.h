#ifndef CONSTANT_H
#define CONSTANT_H

inline constexpr int SAMPLE_RATE = 22050;

inline constexpr int SEMITON_PER_OCTAVE = 12;

inline constexpr int NOTES_BINS_PER_SEMITONE = 1;

inline constexpr int CONTOURS_BINS_PER_SEMITONE = 3;

inline constexpr int ANNOTATIONS_N_SEMITONES = 88;

inline constexpr float MIN_FREQ = 27.5;

inline constexpr int FFT_HOP = 256;

inline constexpr int N_FFT = 8 * FFT_HOP;

inline constexpr int N_BINS_NOTE = ANNOTATIONS_N_SEMITONES * NOTES_BINS_PER_SEMITONE;

inline constexpr int N_BINS_CONTOUR = ANNOTATIONS_N_SEMITONES * CONTOURS_BINS_PER_SEMITONE;

inline constexpr int N_HARMONICS = 8;

#endif // CONSTANT_H