#!/bin/sh

# Change directory (DO NOT CHANGE!)
repoDir=$(dirname "$(realpath "$0")")
echo $repoDir
cd $repoDir

# Recompile if necessary (DO NOT CHANGE!)
mkdir -p build
cd build
cmake ..
make -j4

nvidia-smi

# Change sequence length and dataset here. Default is 10^6. Change it upto 10^8 but at 10^8 the sequential algorithm will run slower
./suffixArray 1000000 ../data/droYak2_new_sanitised.fa

# To use nvprof, comment out the above line and use this command instead
# nvprof ./suffixArray 1000000 ../data/droYak2_new_sanitised.fa