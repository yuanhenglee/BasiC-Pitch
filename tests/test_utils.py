import numpy as np

def get_audio(shorten=False):
    import librosa
    sig, _  = librosa.load("data/Undertale-Megalovania.wav", sr=22050, mono=True)
    if shorten:
        sig = sig[:sig.shape[0] // 4]
    return sig

def test_matConversion():
    from BasiCPP_Pitch.utils import testMatConversion

    np_in = np.arange(0, 24).reshape(2, 3, 4)

    mat = testMatConversion(np_in)

    assert np.allclose(mat, np_in)

def test_conv2d():
    from BasiCPP_Pitch.utils import testConv2d
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

def test_windowed_audio():
    from BasiCPP_Pitch.utils import getWindowedAudio

    np_in = get_audio(shorten=True)
    print(np_in.shape)
    out = getWindowedAudio(np_in)
    np_out = np.array(out)
    print(np_out.shape)

    import warnings
    warnings.simplefilter("ignore")
    with warnings.catch_warnings():
        import tensorflow as tf
        np_in = np.concatenate([np.zeros(int(30*256/2)), np_in])
        gold = tf.signal.frame(np_in, 43844, 43844-30*256, pad_end=True, pad_value=0).numpy()
        print(gold.shape)
    assert np.allclose(np_out, gold)

if __name__ == "__main__":
    # test_matConversion()
    # test_conv2d()
    test_windowed_audio()
