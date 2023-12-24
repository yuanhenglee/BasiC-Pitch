import numpy as np

C = 3
H = 5
W = 5
K_H = 3
K_W = 3
stride = 1
out_H = (H - K_H) // stride + 1
out_W = (W - K_W) // stride + 1

def get_input_img():
    return [ np.arange(H*W).reshape(H,W) for _ in range(C) ]

def get_output_col():
    return np.arange(C * out_H * out_W ).reshape(C, out_H*out_W)

def test_im2col():
    from BasiCPP_Pitch.nnUtils import im2col
    x = get_input_img()
    y = im2col(x, out_H, out_W, K_H, K_W, stride)
    print(x)
    print(y)

def test_col2im():
    from BasiCPP_Pitch.nnUtils import im2col, col2im
    x = get_output_col()
    y = col2im(x, out_H, out_W)
    print(x)
    print(y)

def test_conv2d():
    from BasiCPP_Pitch.nnUtils import testConv2d
    import warnings
    warnings.simplefilter("ignore")
    with warnings.catch_warnings():
        import tensorflow as tf

    np_in = np.arange(-6.0, 6.0).reshape(3, 4)
    kernel = np.arange(-4.0, 5.0).reshape(3, 3)
    stride = 1

    np_out = testConv2d(np_in, kernel, stride)

    gold = tf.keras.layers.Conv2D(
        1, 
        3, 
        strides=stride, 
        padding='same', 
        kernel_initializer=tf.keras.initializers.Constant(kernel)
    )(np_in.reshape(1, 3, 4, 1)).numpy().squeeze()

    assert np.allclose(np_out, gold)


if __name__ == "__main__":
    # test_im2col()
    test_col2im()