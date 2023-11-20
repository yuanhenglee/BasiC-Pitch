#!/bin/bash

./clean.sh

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
cmake ..
make
cd ..
cp lib/*.so tests/