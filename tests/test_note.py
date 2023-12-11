import numpy as np

def test_infered_onsets():
    from BasiCPP_Pitch.note import getInferedOnsets

    Yn = np.random.rand(10, 5)
    Yo = np.random.rand(10, 5)

    onsets = getInferedOnsets(Yo, Yn)
    print(onsets.shape)
    # print(onsets)

    import warnings
    warnings.simplefilter("ignore")
    with warnings.catch_warnings():
        from basic_pitch.note_creation import get_infered_onsets
        gold = get_infered_onsets(Yo, Yn)
        print(gold.shape)
        # print(gold)

    assert np.allclose(onsets, gold)

if __name__ == "__main__":
    test_infered_onsets()