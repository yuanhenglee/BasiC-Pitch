#!/bin/bash

./clean.sh

if [ ! -d "src/ext/eigen-3.3.7" ] || [ ! -d "src/ext/fftw-3.3.10" ]; then
    cur_wd=$(pwd)
    cd src/ext
    tar -zxf eigen-3.3.7.tar.gz  # eigen do not need to install
    tar -zxf fftw-3.3.10.tar.gz
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