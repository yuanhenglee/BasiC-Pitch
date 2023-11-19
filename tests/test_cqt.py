def test_cqt():
    import BasiCPP_Pitch
    import numpy as np

    length = 1000
    n_channels = 2
    np_arr = np.random.rand(n_channels, length).astype(np.float32)
    param = BasiCPP_Pitch.CQParams(44100, 12, 0.0, 3520, 84)
    cqt = BasiCPP_Pitch.CQT(np_arr, param)
    print(cqt)

if __name__ == "__main__":
    test_cqt()

