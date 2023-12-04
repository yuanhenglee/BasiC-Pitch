import numpy as np

def test_cnn():
    import BasiCPP_Pitch

    cnn = BasiCPP_Pitch.CNN("Contour")
    target = ["8 Conv2D (3x39)", "1 Conv2D (5x5)"]
    for t in target:
        assert t in str(cnn)


if __name__ == '__main__':
    test_cnn()