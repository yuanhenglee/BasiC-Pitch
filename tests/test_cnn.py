import numpy as np


def test_weight():
    import BasiCPP_Pitch

    cnn = BasiCPP_Pitch.CNN("Contour")

    weights = cnn.getFirstKernel()
    print(weights.shape)

    import json

    with open("model/cnn_contour_model.json", "r") as f:
        j = json.load(f)
        gold = np.array(j["layers"][0]["weights"][0])
        # gold = np.array(j["layers"]["weights"])
        print(gold.shape)
        gold = gold.transpose(2, 3, 0, 1)
        print(gold.shape)

    assert np.allclose(weights, gold[0, 0, :, :].squeeze())


if __name__ == "__main__":
    test_weight()
