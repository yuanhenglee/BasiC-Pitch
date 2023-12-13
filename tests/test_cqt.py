import math
import timeit
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_cqt( results, filename='cqt.png' ):
    plt.figure(figsize=(12, 6))
    for i, (name, arr) in enumerate(results.items()):
        plt.subplot(len(results), 1, i + 1)
        plt.title(name)
        plt.imshow(arr, cmap='coolwarm', vmin=0, vmax=1)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), filename))

def plot_hs( results, filename='hs.png' ):
    plt.figure(figsize=(12, 6))
    for i, (name, arr) in enumerate(results.items()):
        plt.subplot(len(results), 1, i + 1)
        plt.title(name)
        arr = np.concatenate(arr, axis=0)
        plt.imshow(arr, cmap='coolwarm', vmin=0, vmax=1)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), filename))

def get_audio(shorten=False):
    import librosa
    # sig, _ = librosa.load(librosa.ex('brahms'), sr=22050)
    sig, _  = librosa.load("data/Undertale-Megalovania.wav", sr=22050, mono=True)
    # shortening the signal for quicker testing
    if shorten:
        sig = sig[:sig.shape[0] // 4]
    return sig

def test_harmonic_stacking(vis = False):
    import BasiCPP_Pitch
    
    np_arr = get_audio(shorten=False)
    np_arr = np.ascontiguousarray(np_arr, dtype=np.float32)
    t = BasiCPP_Pitch.CQ()
    res = t.harmonicStacking(np_arr, batch_norm = False).transpose(0, 2, 1)
    print(res.shape)

    # supress warning
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from basic_pitch.nn import HarmonicStacking
        from basic_pitch.models import get_cqt
        import tensorflow as tf
        audio = tf.expand_dims(np_arr, 0)
        audio = tf.expand_dims(audio, -1)
        x = get_cqt(audio, 8, False)
        x = HarmonicStacking(
            3,
            [0.5] + list(range(1, 8)),
            264
        )(x)
        gold = x.numpy().squeeze().transpose(2, 1, 0)
    print(gold.shape)

    assert np.allclose(res, gold, atol=1e-3)

    if vis:
        plot_hs({
            'Baseline': gold,
            'Ours': res, 
        })


def test_cqt(vis = False):
    import BasiCPP_Pitch

    np_arr = get_audio(shorten=True)
    np_arr = np.ascontiguousarray(np_arr, dtype=np.float32)

    t = BasiCPP_Pitch.CQ()
    res = t.computeCQT(np_arr, batch_norm=False)
    print(res.shape)

    # supress warning
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from basic_pitch.models import get_cqt
        import tensorflow as tf
        audio = tf.expand_dims(np_arr, 0)
        audio = tf.expand_dims(audio, -1)
        gold = get_cqt(audio, 8, False).numpy().squeeze().T
    print(gold.shape)

    assert np.allclose(res, gold, atol=1e-3)
    
    if vis:
        plot_cqt({
            'Baseline': gold,
            'Ours': res, 
        })


if __name__ == "__main__":
    # test_cqt(vis = True)
    test_harmonic_stacking( vis=True )

