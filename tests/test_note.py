import numpy as np

def get_audio(shorten=False):
    import librosa
    sig, _  = librosa.load("data/Undertale-Megalovania.wav", sr=22050, mono=True)
    # shortening the signal for quicker testing
    if shorten:
        sig = sig[:sig.shape[0] // 4]
    return sig

def test_infered_onsets():
    from BasiCPP_Pitch.note import getInferedOnsets

    audio = get_audio(shorten=False)

    import warnings
    warnings.simplefilter("ignore")
    with warnings.catch_warnings():
        from basic_pitch.inference import predict
        from basic_pitch.note_creation import get_infered_onsets
        model_output, _, _ = predict("data/Undertale-Megalovania.wav", melodia_trick = False)
        Yo = model_output['onset']
        Yn = model_output['note']
        gold = get_infered_onsets(Yo, Yn)

    onsets = getInferedOnsets(Yo, Yn)

    assert np.allclose(onsets, gold)

def test_model_output2note():
    # from BasiCPP_Pitch import amtModel
    from BasiCPP_Pitch.note import modelOutput2Notes
    audio = get_audio(shorten=False)

    import warnings
    warnings.simplefilter("ignore")
    with warnings.catch_warnings():
        from basic_pitch.inference import predict
        model_output, midi_data, gold = predict("data/Undertale-Megalovania.wav", melodia_trick = False)
        print(len(gold))

    Yp, Yn, Yo = model_output['contour'], model_output['note'], model_output['onset']
    # print(type(Yp), type(Yn), type(Yo))
    # print(Yp.shape, Yn.shape, Yo.shape)
    notes = modelOutput2Notes( Yp, Yn, Yo, True )
    print(len(notes))

    # assert len(notes) == len(gold)
    # for i in range(5):
    #     print(notes[i])
    #     print(gold[i])
    # for note, gold_note in zip(notes, gold):
    #     assert note.pitch == gold_note.pitch
    #     assert note.start == gold_note.start
    #     assert note.end == gold_note.end
    #     assert note.amplitude == gold_note.amplitude



if __name__ == "__main__":
    # test_infered_onsets()
    test_model_output2note()