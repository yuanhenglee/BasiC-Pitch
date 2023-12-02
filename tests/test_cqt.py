import math
import timeit

def plot_cqt( results ):
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    plt.figure(figsize=(12, 6))
    for i, (name, arr) in enumerate(results.items()):
        plt.subplot(len(results), 1, i + 1)
        plt.title(name)
        # plt.imshow(arr, cmap='coolwarm', origin='lower', aspect='auto', interpolation='nearest')
        arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
        plt.pcolor(arr, cmap='coolwarm', vmin=0, vmax=1)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'vis.png'))

def get_audio(shorten=False):
    import librosa
    import numpy as np
    # sig, _ = librosa.load(librosa.ex('brahms'), sr=22050)
    sig, _  = librosa.load("data/Undertale-Megalovania.wav", sr=22050, mono=True)
    # shortening the signal for quicker testing
    if shorten:
        sig = sig[:sig.shape[0] // 4]
    return sig

def visualize_kernel():
    import BasiCPP_Pitch
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    param = BasiCPP_Pitch.CQParams()
    print(param)
    t = BasiCPP_Pitch.CQ(param)
    res = t.getKernel()
    print(res.shape)

    np.save('kernel.npy', res)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Real')
    plt.imshow(np.real(res), cmap='coolwarm', origin='lower', aspect='auto', interpolation='nearest')
    plt.subplot(1, 2, 2)
    plt.title('Imag')
    plt.imshow(np.imag(res), cmap='coolwarm', origin='lower', aspect='auto', interpolation='nearest')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'kernel.png'))

def test_speed():
    import BasiCPP_Pitch
    import numpy as np
    import librosa
    from sklearn.preprocessing import minmax_scale

    np_arr = get_audio(shorten=False)
    np_arr = np.ascontiguousarray(np_arr, dtype=np.float32)

    param = BasiCPP_Pitch.CQParams()
    t = BasiCPP_Pitch.CQ(param)

    def compute_librosa():
        librosa_res = librosa.cqt(np_arr,
            sr=22050,
            hop_length=256,
            fmin=27.5,
            n_bins=88,
            bins_per_octave=12,
        )
        librosa_res = np.abs(librosa_res)
        librosa_res = minmax_scale(librosa_res, axis=1)

    def compute_ours():
        t.computeCQT(np_arr)

    init_time = timeit.timeit(lambda: BasiCPP_Pitch.CQ(param), number=10)
    librosa_time = timeit.timeit(compute_librosa, number=10)
    our_time = timeit.timeit(compute_ours, number=3)
    speedup = librosa_time / (our_time + init_time)

    print('Init time: {:.3f} ms'.format(init_time * 1000))
    print('Librosa time: {:.3f} ms'.format(librosa_time * 1000))
    print('Ours compute time: {:.3f} ms'.format(our_time * 1000))
    print('Ours total time: {:.3f} ms'.format((init_time + our_time) * 1000))
    print('Speedup: {:.3f}x'.format(speedup))
    assert speedup > 1e-1
    
# def test_vs_librosa(vis = False):
#     import BasiCPP_Pitch
#     import numpy as np
#     import librosa
#     import time
#     from sklearn.preprocessing import normalize, minmax_scale

#     np_arr = get_audio(shorten=True)
#     np_arr = np.ascontiguousarray(np_arr, dtype=np.float32)

#     librosa_res = librosa.cqt(np_arr,
#         sr=22050,
#         hop_length=256,
#         fmin=27.5,
#         n_bins=309,
#         bins_per_octave=12 * 3,
#     )
#     librosa_res = np.abs(librosa_res)
#     librosa_res = minmax_scale(librosa_res, axis=1)

#     param = BasiCPP_Pitch.CQParams()
#     print(param)
#     t = BasiCPP_Pitch.CQ(param)
#     res = t.computeCQT(np_arr)
#     res = minmax_scale(res, axis=1)

#     h, w = min(librosa_res.shape[0], res.shape[0]), min(librosa_res.shape[1], res.shape[1])
#     diff = np.abs(librosa_res[:h, :w] - res[:h, :w])

#     print('Mean diff: {:.3f}'.format(np.mean(diff)))
#     # assert np.mean(diff) < 1e-1

#     if vis:
#         # visualize
#         plot_cqt({
#             'librosa': librosa_res,
#             'Ours': res,
#             'Diff': diff,
#         })

def visualize_harmonic_stacking():
    import BasiCPP_Pitch
    import numpy as np
    
    np_arr = get_audio()
    np_arr = np.ascontiguousarray(np_arr, dtype=np.float32)
    param = BasiCPP_Pitch.CQParams()
    print(param)
    t = BasiCPP_Pitch.CQ(param)
    res = t.harmonicStacking(np_arr)
    print(res.shape)
    # print(np.max(res), np.min(res))
    # res = (res - np.min(res)) / (np.max(res) - np.min(res))

    print(res[:, 30:50, -5:])
    # hs = {i:res[i] for i in range(res.shape[0])}
    # plot_cqt(hs)

def test_lowpass_filter():
    import BasiCPP_Pitch
    import numpy as np

    np_arr = get_audio(shorten=False)
    np_arr = np.ascontiguousarray(np_arr, dtype=np.float32)

    lf = BasiCPP_Pitch.createLowPassFilter()

    res = BasiCPP_Pitch.downsamplingByN(np_arr, lf, 2)

    print(res.shape)
    print(res[:, -10:])

def visualize_cqt():
    import BasiCPP_Pitch
    import numpy as np
    import time

    np_arr = get_audio(shorten=False)
    np_arr = np.ascontiguousarray(np_arr, dtype=np.float32)

    start = time.time()
    param = BasiCPP_Pitch.CQParams()
    # print(param)
    t = BasiCPP_Pitch.CQ(param)
    res = t.computeCQT(np_arr)
    print('C++ time: {:.3f}'.format(time.time() - start))
    print(res.shape)
    print(res[:5, :5])

    # plot_cqt({
    #     'CQT': res, 
    #     # 'CQT_real': res.real,
    #     # 'CQT_imag': res.imag,
    # })

    


if __name__ == "__main__":
    # test_speed()
    # visualize_kernel()
    visualize_harmonic_stacking()
    # test_lowpass_filter()
    # visualize_cqt()

