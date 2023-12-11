import numpy as np

def get_audio(shorten=False):
    import librosa
    sig, _  = librosa.load("data/Undertale-Megalovania.wav", sr=22050, mono=True)
    if shorten:
        sig = sig[:sig.shape[0] // 4]
    return sig

def test_amtModelCQ():
    import BasiCPP_Pitch

    bp_model = BasiCPP_Pitch.amtModel()
    t = bp_model.getCQ()

    lf = t.getFilter()
    gold = np.load('model/lowpass_filter.npy')

    assert np.allclose(lf, gold)

    kernel = t.getKernel()
    gold = np.load('model/kernel.npy')

    assert np.allclose(kernel, gold)

def plot_hm( datas ):
    import matplotlib.pyplot as plt
    plt.figure(figsize=( 8*2, 6 ))
    for i, prefix in enumerate(["Baseline", "Ours"]):
        for j, suffix in enumerate(["contour", "note", "onset"]):
            plt.subplot(3, 2, i+2*j + 1)
            plt.title(prefix + " " + suffix)
            v = datas[prefix + " " + suffix]
            v = (v - v.min()) / (v.max() - v.min())
            plt.imshow(v.T, cmap='coolwarm', vmin=0, vmax=1, aspect='auto', origin='lower')
            # plt.pcolor(v.T, cmap='coolwarm', vmin=0, vmax=1)
    plt.tight_layout()
    plt.savefig("./tests/model_output.png")

def print_distribution( datas ):
    for k, v in datas.items():
        print(k, v.shape)
        print("min", v.min())
        print("max", v.max())
        print("mean", v.mean())
        print("std", v.std())
        print("")

def test_inference(vis = False):
    import BasiCPP_Pitch
    import time

    plot_dict = {}

    bp_model = BasiCPP_Pitch.amtModel()
    np_arr = get_audio()
    start_time = time.time()
    # Yp, Yn, Yo = bp_model.inference(np_arr)
    notes = bp_model.transcribeAudio(np_arr)
    Yp, Yn, Yo = bp_model.getOutput()
    print(f"Elapsed time: {time.time() - start_time}")

    plot_dict['Ours contour'] = np.array(Yp)
    plot_dict['Ours note'] = np.array(Yn)
    plot_dict['Ours onset'] = np.array(Yo)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        start_time = time.time()
        from basic_pitch.inference import run_inference
        from basic_pitch import ICASSP_2022_MODEL_PATH
        from tensorflow import saved_model
        model = saved_model.load(str(ICASSP_2022_MODEL_PATH))
        gold = run_inference("data/Undertale-Megalovania.wav", model)
    print(f"Elapsed time: {time.time() - start_time}")
    for k, v in gold.items():
        plot_dict["Baseline " + k] = v

    print_distribution(plot_dict)

    for output in ["onset", "contour", "note"]:
        assert plot_dict["Baseline " + output].shape == plot_dict["Ours " + output].shape

    if vis:
        plot_hm(plot_dict)


if __name__ == "__main__":
    test_inference(vis=True)
    # test_amtModelCQ()