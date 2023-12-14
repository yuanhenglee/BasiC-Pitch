#include "typedef.h"
#include "amtModel.h"
#include "loader.h"
#include <iostream>
#include <string>

int main(int argc, char** argv) {

    bool benchmark = false;
    for ( int i = 0 ; i < argc ; i++ ) {
        if ( std::string(argv[i]) == "-b" || std::string(argv[i]) == "--benchmark" ) {
            benchmark = true;
        }
    }

    auto audio = getExampleAudio();

    auto model = amtModel();

    auto notes = model.transcribeAudio(audio);

    return 0;
}