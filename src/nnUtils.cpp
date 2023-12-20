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
        ;
            result(i, j) = (
                padded_x.block(i , j * stride, filter_kernel.rows(), filter_kernel.cols())\
                .cwiseProduct(filter_kernel)
            ).sum();
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

inline bool valid_idx( int idx, int max_idx ) {
    return idx >= 0 && idx < max_idx;
}

// shape of input: (C, HW)
Matrixf im2col( const Matrixf& input,
    const int n_frames_in, const int n_features_in, const int n_frames_out, const int n_features_out,
    const int kernel_height, const int kernel_width, const int stride) {

    int n_filters_in = input.rows();
    int pad_height = padLength(n_frames_in, kernel_height, 1, n_frames_out);
    int pad_width = padLength(n_features_in, kernel_width, stride, n_features_out);

    Matrixf output(n_filters_in * kernel_height * kernel_width, n_frames_out * n_features_out);
    // iterate by the order of output
    for ( int c = 0 ; c < n_filters_in ; c++ ) {
        for ( int kernel_row_idx = 0 ; kernel_row_idx < kernel_height ; kernel_row_idx++ ) {
            for ( int kernel_col_idx = 0 ; kernel_col_idx < kernel_width ; kernel_col_idx++ ) {
                int output_row_idx = c * kernel_height * kernel_width + kernel_row_idx * kernel_width + kernel_col_idx;
                int input_row_idx = -(pad_height / 2) + kernel_row_idx;

                for ( int i = 0 ; i < n_frames_out ; i++ ) {
                    if ( !valid_idx(input_row_idx, n_frames_in) ) {
                        // set all values in this row to 0
                        output.row(output_row_idx).setZero();
                        input_row_idx += 1;
                        continue;
                    }

                    int input_col_idx = -(pad_width / 2) + kernel_col_idx;
                    for ( int j = 0 ; j < n_features_out ; j++ ) {
                        int output_col_idx = i * n_features_out + j;
                        if ( !valid_idx(input_col_idx, n_features_in) ) {
                            output(output_row_idx, output_col_idx) = 0;
                        }
                        else {
                            output(output_row_idx, output_col_idx) = input(c, input_row_idx * n_features_in + input_col_idx);
                        }
                        input_col_idx += stride;
                    }
                    input_row_idx += 1;
                }

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