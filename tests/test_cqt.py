def plot_cqt( results ):
    import matplotlib.pyplot as plt
    import os
    plt.figure(figsize=(12, 6))
    for i, (name, arr) in enumerate(results.items()):
        plt.subplot(len(results), 1, i + 1)
        plt.title(name)
        # plt.imshow(arr, cmap='coolwarm', origin='lower', aspect='auto', interpolation='nearest')
        plt.pcolor(arr, cmap='coolwarm', vmin=0, vmax=1)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'vis.png'))

def get_audio():
    import librosa
    import numpy as np
    sig, _ = librosa.load(librosa.ex('brahms'), sr=16000)
    print(sig.shape)
    print(np.max(sig), np.min(sig))
    return sig

def test_cqt():
    import BasiCPP_Pitch
    import numpy as np
    import librosa
    from sklearn.preprocessing import normalize, minmax_scale

    np_arr = get_audio()
    np_arr = np.ascontiguousarray(np_arr, dtype=np.float32)
    
    librosa_res = librosa.cqt(np_arr,
        sr=22050,
        hop_length=256,
        fmin=27.5,
        n_bins=88,
        bins_per_octave=12,
    )
    librosa_res = np.abs(librosa_res)
    # librosa_res = librosa.util.normalize(librosa_res, axis=1, norm = 0)
    librosa_res = minmax_scale(librosa_res, axis=0)
    print(librosa_res.shape)
    print(np.max(librosa_res), np.min(librosa_res))
    print(librosa_res[:5, :3])

    param = BasiCPP_Pitch.CQParams()
    print(param)
    t = BasiCPP_Pitch.CQ(param)
    res = t.compute_cqt(np_arr)
    res = np.abs(res)
    # # res = librosa.util.normalize(res, axis=1, norm = 0)
    # # res = normalize(res, axis=1, norm = 'l1')
    # # res = np.clip(res, 0, 1)
    print(res.shape)
    print(np.max(res), np.min(res))
    res = minmax_scale(res, axis=1, feature_range=(0, 1))
    print( res[:5, :3])


    # visualize
    plot_cqt({
        'librosa': librosa_res,
        'Ours': res,
    })

if __name__ == "__main__":
    test_cqt()

