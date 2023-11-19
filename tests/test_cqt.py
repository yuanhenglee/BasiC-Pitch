def test_cqt():
    import BasiCPP_Pitch
    import numpy as np

    length = 10
    n_channels = 2
    np_arr = np.random.rand(n_channels, length).astype(np.float32)
    param = BasiCPP_Pitch.CQParams(16000, 12, 32, 8000, 2)
    # cqt = BasiCPP_Pitch.CQT(np_arr, param)
    t = BasiCPP_Pitch.CQ(param)
    res = t.compute_cqt(np_arr)
    print(res)

if __name__ == "__main__":
    test_cqt()

