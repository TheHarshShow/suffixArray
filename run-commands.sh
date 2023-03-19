#!/bin/sh

# Change directory (DO NOT CHANGE!)
repoDir=$(dirname "$(realpath "$0")")
echo $repoDir
cd $repoDir

# Recompile if necessary (DO NOT CHANGE!)
mkdir -p build
cd build
cmake  -DTBB_DIR=${repoDir}/../oneTBB-2019_U9  -DCMAKE_PREFIX_PATH=${repoDir}/../oneTBB-2019_U9/cmake ..
make -j4
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PWD}/tbb_cmake_build/tbb_cmake_build_subdir_release

nvidia-smi
## Basic run to map the first 40 reads in the reads.fa in batches of 10 reads
## HINT: may need to change values for the assignment tasks. You can create a sequence of commands
# nvprof ./readMapper --reference ../data/reference.fa --reads ../data/reads.fa  --maxReads 150000 --batchSize 1000 --numThreads 8
nvprof ./suffixArray