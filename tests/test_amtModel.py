import numpy as np

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

if __name__ == "__main__":
    test_amtModelCQ()