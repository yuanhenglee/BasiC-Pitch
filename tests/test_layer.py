import numpy as np

def test_relu():
    from BasiCPP_Pitch import layer

    np_in = np.arange(-12, 12).reshape(2, 3, 4)

    relu = layer.ReLU()

    np_out = relu.forward(np_in)

    print(np_out)

if __name__ == '__main__':
    test_relu()

    
    