
def get_audio():
    import librosa
    sig, _ = librosa.load(librosa.ex('brahms'), sr=16000)
    return sig

def test_cqt():
    import BasiCPP_Pitch
    import numpy as np
    import librosa

    np_arr = get_audio()
    np_arr = np.ascontiguousarray(np_arr, dtype=np.float32)
    
    librosa_res = librosa.cqt(np_arr,
        sr=16000,
        hop_length=640,
        fmin=32,
        n_bins=84,
        bins_per_octave=12,
    )
    librosa_res = np.abs(librosa_res)
    librosa_res = librosa.util.normalize(librosa_res)

    param = BasiCPP_Pitch.CQParams(
        16000,  # sample rate
        12,  # bins_per_octave
        32,  # fmin
        4096,  # fmax
        0.04,  # hop length
    )
    print(param)
    t = BasiCPP_Pitch.CQ(param)
    res = t.compute_cqt(np_arr)
    res = np.abs(res)
    res = librosa.util.normalize(res)


    # visualize
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.title("librosa")
    plt.imshow(librosa_res, cmap='coolwarm', origin='lower', aspect='auto', interpolation='nearest')
    plt.subplot(3, 1, 2)
    plt.title("BasiCPP")
    plt.imshow(res, cmap='coolwarm', origin='lower', aspect='auto', interpolation='nearest')
    plt.subplot(3, 1, 3)
    plt.title("audio")
    plt.plot(np_arr)    

    plt.tight_layout()
    plt.savefig("test_cqt.png")


if __name__ == "__main__":
    test_cqt()

