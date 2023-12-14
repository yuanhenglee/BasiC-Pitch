#!/bin/bash

python="OFF"
exe="OFF"
test="OFF"
while getopts ":pet" "opt"; do
    case ${opt} in
        "p" ) python="ON"
            ;;
        "e" ) exe="ON"
            ;;
        "t" ) test="ON"
            ;;
        * ) echo "Usage: cmd [-p] [-e]"
            echo "  -p: build python module"
            echo "  -e: build executable"
            echo "  -t: run tests, only valid when python module is built"
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

if [ ! -d "src/ext/fftw-3.3.10" ]; then
    if [ ! -f "src/ext/fftw-3.3.10.tar.gz" ]; then
        wget http://www.fftw.org/fftw-3.3.10.tar.gz -P src/ext
    fi
    cur_wd=$(pwd)
    cd src/ext
    tar -zxf fftw-3.3.10.tar.gz
    rm fftw-3.3.10.tar.gz
    cd fftw-3.3.10
    ./configure --enable-shared --enable-float  -prefix=${PWD}/build 
    make CFLAGS="-fPIC"
    make install
    cd ${cur_wd}
fi

mkdir build
cd build
# cmake .. -D BUILD_PY=ON -D BUILD_EXE=ON
cmake .. -D BUILD_PY=${python} -D BUILD_EXE=${exe}
make
cd ..
if [ ${python} == "ON" ]; then
    cp build/lib/*.so python/
    cp lib/*.so tests/
    if [ ${test} == "ON" ]; then
        pytest
    fi
fi