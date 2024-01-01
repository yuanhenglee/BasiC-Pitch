#include "nnUtils.h"
#include "typedef.h"

inline int padLength(int input_length, int filter_length, int stride, int output_length) {
    return (output_length-1) * stride + filter_length - input_length;
}

int computeNFeaturesOut(int n_features_in, int kernel_size_feature, int stride) {
    // padding == "same"
    float f = static_cast<float>(n_features_in) / static_cast<float>(stride);
    return std::ceil(f);
}

// NOTE: use VALID padding as default
Vectorf conv1d( Vectorf &x, Vectorf &filter_kernel, int stride ) {
    std::vector<float> result;
    for ( int i = 0 ; i + filter_kernel.size() <= x.size() ; i += stride ) {
        Vectorf temp = x.segment(i, filter_kernel.size());
        result.push_back(temp.dot(filter_kernel));
    }
    return Eigen::Map<Vectorf>(result.data(), result.size());
}

// NOTE: use SAME padding as default
// x.shape = (n_samples, n_features_in)
// NOTE: Since we are dealing with audio signals, the number of samples remains the same
Matrixf conv2d( const Matrixf &x, const Matrixf &filter_kernel, int stride ) {
    int n_samples_out = x.rows();
    int n_features_out = computeNFeaturesOut(x.cols(), filter_kernel.cols(), stride);
    int pad_height = padLength(x.rows(), filter_kernel.rows(), 1, n_samples_out);
    int pad_width = padLength(x.cols(), filter_kernel.cols(), stride, n_features_out);
    Matrixf result(n_samples_out, n_features_out);
    Matrixf padded_x = Matrixf::Zero(x.rows() + pad_height, x.cols() + pad_width);
    padded_x.block(pad_height / 2, pad_width / 2, x.rows(), x.cols()) = x;
    
    // Matrixf temp(filter_kernel.rows(), filter_kernel.cols());
    for ( int i = 0 ; i < n_samples_out ; i++ ) {
        for ( int j = 0 ; j < n_features_out ; j++ ) {
            result(i, j) = (
                padded_x.block(i , j * stride, filter_kernel.rows(), filter_kernel.cols()).cwiseProduct(filter_kernel)
            ).sum();
            // forloop version
            // for ( int k = 0 ; k < filter_kernel.rows() ; k++ ) {
            //     for ( int l = 0 ; l < filter_kernel.cols() ; l++ ) {
            //         result(i, j) += padded_x(i + k, j * stride + l) * filter_kernel(k, l);
            //     }
            // }
        }
    }    

    return result;
}

Vectorf reflectionPadding(const Vectorf &x, int pad_length) {
    Vectorf padded_x = Vectorf::Zero(x.size() + 2 * pad_length);
    padded_x.segment(pad_length, x.size()) = x;
    for ( int i = 0 ; i < pad_length ; i++ ) {
        padded_x[i] = x[pad_length - i];
        padded_x[padded_x.size() - 1 - i] = x[x.size() - 1 - pad_length + i];
    }
    return padded_x;
}

// shape of input: (H, W)
Matrixf im2col( const VecMatrixf& input, int n_frames_out, int n_features_out, int kernel_height, int kernel_width, int stride) {
    int n_filters_in = input.size();
    int n_frames_in = input[0].rows();
    int n_features_in = input[0].cols();
    int pad_height = padLength(n_frames_in, kernel_height, 1, n_frames_out);
    int pad_width = padLength(n_features_in, kernel_width, stride, n_features_out);
    Matrixf padded_input = Matrixf::Zero(n_frames_in + pad_height, n_features_in + pad_width);
    Matrixf output = Matrixf::Zero(n_filters_in * kernel_height * kernel_width, n_frames_out * n_features_out);
    for ( size_t i = 0 ; i < input.size() ; i++ ) {
        padded_input.block(pad_height / 2, pad_width / 2, n_frames_in, n_features_in) = input[i];
        for ( size_t j = 0 ; j < n_frames_out ; j++ ) {
            for ( size_t k = 0 ; k < n_features_out ; k++ ) {
                size_t col_idx = j * n_features_out + k;
                Matrixf target_block = padded_input.block(j, k * stride, kernel_height, kernel_width);
                output.col(col_idx).segment(i * kernel_height * kernel_width, kernel_height * kernel_width) = Eigen::Map<Vectorf>(target_block.data(), target_block.size());
            }
        }
    }
    return output;
}

// shape of input: (n_filters_out, n_frames_out * n_features_out)
// shape of output: (n_filters_out, n_frames_out, n_features_out)
VecMatrixf col2im( const Matrixf& input, int n_frames_out, int n_features_out ) {
    int n_filters_out = input.rows();
    VecMatrixf output(n_filters_out, Matrixf::Zero(n_frames_out, n_features_out));
    for ( size_t i = 0 ; i < n_filters_out ; i++ ) {
        Matrixf row = input.row(i);
        output[i] = Eigen::Map<Matrixf>(row.data(), n_frames_out, n_features_out);
    }
    return output;
}