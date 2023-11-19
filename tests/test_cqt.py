def test_cqt():
    import BasiCPP_Pitch
    import numpy as np

    np_arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    param = BasiCPP_Pitch.CQParams(44100, 12, 0.0, 3520, 84)
    cqt = BasiCPP_Pitch.CQT(np_arr, param)
    print(cqt)

if __name__ == "__main__":
    test_cqt()

