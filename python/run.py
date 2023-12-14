import librosa
from midi_utils import note2midi

# Specify the input audio file path
audio_file_path = "data/Undertale-Megalovania.wav"
# audio_file_path = "../data/Undertale-Megalovania.wav"

# Specify the output MIDI file path
midi_file_path = "data/output/Undertale-Megalovania.mid"

def getExampleAudio():
    sig, _  = librosa.load(audio_file_path, sr=22050, mono=True)
    return sig

def main():
    
    # Load the example audio
    audio = getExampleAudio()

    # Initialize the model
    model = BasiCPP_Pitch.amtModel()

    # Transcribe the audio and generate the MIDI file
    notes = model.transcribeAudio(audio)
    midi = note2midi(notes)
    midi.write(midi_file_path)

if __name__ == "__main__":
    try:
        import BasiCPP_Pitch # Import the BasiCPP Pitch Python module
        main()
    except ImportError: 
        print("BasiCPP Pitch Python module not found. Resolve by using basic_pitch instead.")
        import os 
        os.system(f"basic-pitch {midi_file_path} {audio_file_path}")