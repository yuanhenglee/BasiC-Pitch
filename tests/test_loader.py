import math
import timeit
import numpy as np
import matplotlib.pyplot as plt
import os

def test_kernel(vis = False):
    import BasiCPP_Pitch

    t = BasiCPP_Pitch.CQ()
    res = t.getKernel()
    print(res.shape)

    gold = np.load('model/kernel.npy')

    assert np.allclose(res, gold)

    if vis:
        plt.figure(figsize=(12, 12))
        plt.subplot(2, 2, 1)
        plt.title('Baseline - Real')
        plt.imshow(np.real(gold), cmap='coolwarm', origin='lower', aspect='auto', interpolation='nearest')
        plt.subplot(2, 2, 2)
        plt.title('Ours - Real')
        plt.imshow(np.real(res), cmap='coolwarm', origin='lower', aspect='auto', interpolation='nearest')
        plt.subplot(2, 2, 3)
        plt.title('Baseline - Imag')
        plt.imshow(np.imag(gold), cmap='coolwarm', origin='lower', aspect='auto', interpolation='nearest')
        plt.subplot(2, 2, 4)
        plt.title('Ours - Imag')
        plt.imshow(np.imag(res), cmap='coolwarm', origin='lower', aspect='auto', interpolation='nearest')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), 'kernel.png'))

def test_lowpass_filter(vis = False):
    import BasiCPP_Pitch
    
    t = BasiCPP_Pitch.CQ()
    lf = t.getFilter()

    print(lf.shape)
    gold = np.load('model/lowpass_filter.npy')
    print(gold.shape)


    assert np.allclose(lf, gold)

    if vis:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title('Baseline')
        plt.plot(gold)
        plt.subplot(1, 2, 2)
        plt.title('Ours')
        plt.plot(lf)
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), 'lowpass_filter.png'))

if __name__ == "__main__":
    test_kernel(vis=True)
    test_lowpass_filter(vis=True)

