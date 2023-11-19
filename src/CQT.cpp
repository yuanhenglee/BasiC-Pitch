#include "CQT.h"
#include <iostream>
// #include "kiss_fft.h"

CQParams::CQParams(float sample_rate, int bins_per_octave, float freq_min, float freq_max, int n_bins)
    : sample_rate(sample_rate), bins_per_octave(bins_per_octave), freq_min(freq_min), freq_max(freq_max), n_bins(n_bins) {  
    std::cout<<"CQParams constructor"<<std::endl;
}   

/*
    This function is the entry point of the CQT algorithm for the pybind interface.
    Input:
        audio: a NumPy array of type float and shape (length, n_channels)
        params: a CQParams object
    Output: 
        a NumPy array of type float and shape (length * n_freq_dims, n_channels)
*/
py::array_t<float> constantQTransform(
    py::array_t<float> audio,
    CQParams params
) {

    CQ t(params);
    py::array_t<float> result = t.compute_cqt(audio);

    return result;

}

CQ::CQ(CQParams params) : params(params) {
    runKernel();
}

CQ::~CQ() {
    std::cout<<"CQ destructor"<<std::endl;
}

void CQ::runKernel() {
    std::cout<<"CQ::runKernel()"<<std::endl;
}

py::array_t<float> CQ::compute_cqt(py::array_t<float> audio) {
    std::cout<<"CQ::compute_cqt()"<<std::endl;
    py::buffer_info buf = audio.request();
    int length = buf.shape[0];  
    int n_channels = buf.shape[1];  
    std::cout<<"length: "<<length<<std::endl;
    std::cout<<"n_channels: "<<n_channels<<std::endl;
    // TODO
    //temporarily return the input audio
    py::array_t<float> result = py::array_t<float>(audio);
    return result;
}
