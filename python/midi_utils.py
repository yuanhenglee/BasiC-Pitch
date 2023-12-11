from BasiCPP_Pitch.note import Note
from BasiCPP_Pitch import amtModel
import pretty_midi
from typing import List

def note2midi( notes: List[Note], midi_tempo: float = 120 ) -> pretty_midi.PrettyMIDI:
    """Convert notes to midi file

    Args:
        notes (List[Note]): List of notes
        midi_tempo (float, optional): Tempo of midi file. Defaults to 120.

    Returns:
        pretty_midi.PrettyMIDI: Midi file
    """
    midi = pretty_midi.PrettyMIDI(initial_tempo=midi_tempo)
    instrument = pretty_midi.Instrument(program=0)

    for note in notes:
        note = pretty_midi.Note(
            velocity=round(note.amplitude * 127),
            pitch=note.pitch,
            start=note.start,
            end=note.end
        )
        instrument.notes.append(note)

    midi.instruments.append(instrument)
    return midi

if __name__ == "__main__":
    import librosa

    model = amtModel()
    audio, _  = librosa.load("data/Undertale-Megalovania.wav", sr=22050, mono=True)
    notes = model.transcribeAudio(audio)
    print(len(notes))
    midi = note2midi(notes)
    midi.write("data/output/Undertale-Megalovania.mid")

