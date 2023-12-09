# Specify the input audio file path
audio_file_path = "data/Undertale-Megalovania.wav"
# audio_file_path = "../data/Undertale-Megalovania.wav"

# Specify the output MIDI file path
midi_file_path = "data/output"

try:
    import BasiCPP_Pitch # Import the BasiCPP Pitch Python module
    main() # Run the main function
except ImportError: 
    print("BasiCPP Pitch Python module not found. Resolve by using basic_pitch instead.")
    import os 
    os.system(f"basic-pitch {midi_file_path} {audio_file_path}")


def main():
    # Initialize BasiCPP Pitch
    amt = BasiCPP_Pitch.amtModel()

    # Transcribe the audio and generate the MIDI file
    amt.transcribe_audio(audio_file_path, midi_file_path)