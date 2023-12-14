#include "amtModel.h"
#include "loader.h"

int main(int argc, char** argv) {

    // Load the example audio
    auto audio = getExampleAudio();

    // Initialize the model
    auto model = amtModel();

    // Transcribe the audio
    auto notes = model.transcribeAudio(audio);

    return 0;
}