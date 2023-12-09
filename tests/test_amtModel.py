import numpy as np
import tensorflow as tf
from basic_pitch.models import model

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
    # fig, axs = plt.subplots(len(datas), 1, figsize=(15, 20))
    plt.figure(figsize=( 8*2, len(datas) ))
    for i, (k, v) in enumerate(datas.items()):
        if i < len(datas)//2:
            plt.subplot(len(datas)//2, 2, i*2+1)
        else:
            plt.subplot(len(datas)//2, 2, (i-len(datas)//2)*2+2)
        plt.title(k)
        v = (v - v.min()) / (v.max() - v.min())
        plt.pcolor(v.T, cmap='coolwarm', vmin=0, vmax=1)
    plt.tight_layout()
    plt.savefig("./tests/model_output.png")

def test_inference(vis = False):
    import BasiCPP_Pitch

    plot_dict = {}

    bp_model = BasiCPP_Pitch.amtModel()
    np_arr = get_audio(shorten=True)[:43844]
    bp_model.inference(np_arr)

    audio = np_arr.reshape(1, -1, 1)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        audio = tf.convert_to_tensor(audio, dtype=tf.float32)
        print(audio.shape)
        gold = model()(audio)
    for k, v in gold.items():
        plot_dict["Baseline " + k] = v.numpy().squeeze()

    Yo = bp_model.getYo()
    Yp = bp_model.getYp()
    Yn = bp_model.getYn()
    plot_dict['Ours onset'] = Yo
    plot_dict['Ours contour'] = Yp
    plot_dict['Ours note'] = Yn

    for k, v in plot_dict.items():
        print(k, v.shape)

    for output in ["onset", "contour", "note"]:
        assert plot_dict["Baseline " + output].shape == plot_dict["Ours " + output].shape

    if vis:
        plot_hm(plot_dict)


if __name__ == "__main__":
    test_inference(vis=True)