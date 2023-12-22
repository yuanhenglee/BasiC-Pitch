#include "amtModel.h"
#include "loader.h"
#include <iostream>

void printRunStats() {
    // Eigen number of threads
    std::cout << "Eigen number of threads: " << Eigen::nbThreads() << std::endl;
}

int main(int argc, char** argv) {

    // printRunStats();

    // Load the example audio
    auto audio = getExampleAudio();

    // Initialize the model
    auto model = amtModel();

    // Transcribe the audio
    auto notes = model.transcribeAudio(audio);

    return 0;
}