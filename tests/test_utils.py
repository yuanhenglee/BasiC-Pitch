import numpy as np

def test_matConversion():
    from BasiCPP_Pitch.utils import testMatConversion

    np_in = np.arange(0, 24).reshape(2, 3, 4)

    mat = testMatConversion(np_in)

    assert np.allclose(mat, np_in)

if __name__ == "__main__":
    test_matConversion()