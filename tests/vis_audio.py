
import numpy as np
import matplotlib.pyplot as plt

def get_audio(shorten=False):
    import librosa
    sig, _  = librosa.load("data/Undertale-Megalovania.wav", sr=22050)
    if shorten:
        sig = sig[:sig.shape[0] // 4]
    return sig

def vis_audio( audio ):
    plt.figure(figsize=( 8, 3 ))
    plt.title("Audio")
    plt.plot(audio)
    plt.tight_layout()
    plt.savefig("./tests/audio.png")    

audio = get_audio()
print(audio.shape)
vis_audio(audio)