#!/bin/bash

python="OFF"
exe="OFF"
test="OFF"
gprofile="OFF"
while getopts ":petg" "opt"; do
    case ${opt} in
        "p" ) python="ON" ;;
        "e" ) exe="ON" ;;
        "t" ) test="ON" ;;
        "g" ) gprofile="ON";;
        * ) echo "Usage: cmd [-p] [-e]"
            echo "  -p: build python module"
            echo "  -e: build executable"
            echo "  -t: run tests, only valid when python module is built"
            echo "  -g: enable gprof profiling"
            exit 1
            ;;
    esac
done

# Build both by default
if [ ${python} == "OFF" ] && [ ${exe} == "OFF" ]; then
    python="ON"
    exe="ON"
fi
if [ ${python} == "ON" ]; then
    echo "Building python module"
fi
if [ ${exe} == "ON" ]; then
    echo "Building executable"
fi

./clean.sh
if [ ! -d "src/ext/eigen-3.3.7" ]; then
    if [ ! -f "src/ext/eigen-3.3.7.tar.gz" ]; then
        wget https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz -P src/ext
    fi
    cur_wd=$(pwd)
    cd src/ext
    tar -zxf eigen-3.3.7.tar.gz
    rm eigen-3.3.7.tar.gz
    cd ${cur_wd}
fi

mkdir build
cd build
cmake .. -DBUILD_PY=${python} -DBUILD_EXE=${exe} -DGPROF=${gprofile}
make
cd ..

if [ ${python} == "ON" ]; then
    cp lib/*.so python/
    cp lib/*.so tests/
    if [ ${test} == "ON" ]; then
        pytest
    fi
fi

# if executable is built and gprof is enabled, run the executable and generate profiling report
if [ ${exe} == "ON" ] && [ ${gprofile} == "ON" ]; then
    ./bin/run && gprof bin/run gmon.out > ./log/profiling.txt
fi