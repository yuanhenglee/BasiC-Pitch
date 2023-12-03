import math
import timeit
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_cqt( results ):
    plt.figure(figsize=(12, 6))
    for i, (name, arr) in enumerate(results.items()):
        plt.subplot(len(results), 1, i + 1)
        plt.title(name)
        plt.pcolor(arr, cmap='coolwarm', vmin=0, vmax=1)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'cqt.png'))

def get_audio(shorten=False):
    import librosa
    # sig, _ = librosa.load(librosa.ex('brahms'), sr=22050)
    sig, _  = librosa.load("data/Undertale-Megalovania.wav", sr=22050, mono=True)
    # shortening the signal for quicker testing
    if shorten:
        sig = sig[:sig.shape[0] // 4]
    return sig

def visualize_harmonic_stacking():
    import BasiCPP_Pitch
    
    np_arr = get_audio()
    np_arr = np.ascontiguousarray(np_arr, dtype=np.float32)
    t = BasiCPP_Pitch.CQ()
    res = t.harmonicStacking(np_arr)
    print(res.shape)
    # print(np.max(res), np.min(res))
    # res = (res - np.min(res)) / (np.max(res) - np.min(res))

    print(res[:, 30:50, -5:])
    # hs = {i:res[i] for i in range(res.shape[0])}
    # plot_cqt(hs)

def test_cqt(vis = False):
    import BasiCPP_Pitch

    np_arr = get_audio(shorten=False)
    np_arr = np.ascontiguousarray(np_arr, dtype=np.float32)

    t = BasiCPP_Pitch.CQ()
    res = t.computeCQT(np_arr)
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

    assert np.allclose(res, gold, atol=1e-2)

    if vis:
        plot_cqt({
            'Baseline': gold,
            'Ours': res, 
        })

    


if __name__ == "__main__":
    test_cqt(vis = True)

