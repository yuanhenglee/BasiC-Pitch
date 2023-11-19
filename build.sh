#!/bin/bash
./clean.sh  
mkdir build
cd build
cmake ..
make
cd ..
cp lib/*.so tests/