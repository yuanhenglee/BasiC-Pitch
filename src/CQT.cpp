#include "CQT.h"
#include <iostream>
// #include "kiss_fft.h"

CQParams::CQParams(float sample_rate, int bins_per_octave, float freq_min, float freq_max, int n_bins)
    : sample_rate(sample_rate), bins_per_octave(bins_per_octave), freq_min(freq_min), freq_max(freq_max), n_bins(n_bins) {  
    std::cout<<"CQParams constructor"<<std::endl;
}   

py::array_t<double> constantQTransform(
    py::array_t<double> audio,
    CQParams params
) {
    // Get a pointer to the audio data
    auto ptr = audio.unchecked<1>();

    // CQT implementation
    // TODO
    // int n_octaves = (int)std::ceil(float());

    // temp placeholder
    std::vector<double> result(ptr.shape(0));
    for (ssize_t i = 0; i < ptr.shape(0); i++) {
        result[i] = ptr(i);
    }
#ifdef DEBUG
    std::cout<<params.sample_rate<<std::endl;       
    std::cout<<params.bins_per_octave<<std::endl;
    std::cout<<params.freq_min<<std::endl;
    std::cout<<params.freq_max<<std::endl;
    std::cout<<params.n_bins<<std::endl;
#endif

    // Create a NumPy array from the result and return
    return pybind11::array_t<double>(
        {static_cast<ssize_t>(result.size())}, // Shape
        {sizeof(double)},                      // Strides
        result.data()                          // Pointer to data
    );
}
