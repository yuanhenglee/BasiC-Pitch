#include "typedef.h"
#include "amtModel.h"
#include "loader.h"
#include <iostream>
#include <string>

int main(int argc, char** argv) {

    auto audio = getExampleAudio();

    auto model = amtModel();

    auto notes = model.transcribeAudio(audio);

    return 0;
}