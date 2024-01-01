#! /bin/bash

# topic="Eigen_builtin_omp"
# topic="eigen_no_para"
# topic="omp_mkl"
# topic="omp_forloopconv"
topic="pthreads_inferenceFrame"
for nbThreads in 1 2 3 4 5 6 7 8 9 10 11 12
# for nbThreads in 1
do
    cmd="./bin/run"
    export OMP_NUM_THREADS=$nbThreads
    /usr/bin/time -f "[%C] %e %M %P" $cmd >> ./log/$topic.log 2>&1
done