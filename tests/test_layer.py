import numpy as np

def test_relu():
    from BasiCPP_Pitch import layer

    np_in = np.arange(-12, 12).reshape(2, 3, 4)

    relu = layer.ReLU()

    np_out = relu.forward(np_in)

    gold = np_in.copy()
    gold[gold < 0] = 0

    assert np.allclose(np_out, gold)

def test_sigmoid():
    from BasiCPP_Pitch import layer
    sigmoid = layer.Sigmoid()

    np_in = np.arange(0, 24).reshape(2, 3, 4)
    np_out = sigmoid.forward(np_in)
    print(np_out)

    gold = 1 / (1 + np.exp(-np_in))
    print(gold)

    # assert np.allclose(np_out, gold)

    np_in = np.arange(-12, 12).reshape(2, 3, 4)
    np_out = sigmoid.forward(np_in)
    print(np_out)

    gold = 1 / (1 + np.exp(-np_in))
    print(gold)

    # assert np.allclose(np_out, gold)

if __name__ == '__main__':
    # test_relu()
    test_sigmoid()

    
    