# Parallelizing Suffix Array Construction

## Table of Contents
- [Overview](#overview)
- [Getting Started](#gettingstarted) 
- [How to use](#usage)
- [References](#references)

## <a name="overview"></a> Overview

We propose a parallel implementation for an efficient suffix arrays construction algorithm described by Flick et al [1] on a GPU architecture. We contributed the approach of parallelizing prefix scan across multiple thread blocks in GPU which offered massive parallelism. Our experimental results show up to 670x and 16x speedup over unoptimized CPU implementation as well as with state-of-the-art libdivsufsort library respectively.

## <a name="gettingstarted"></a> Getting Started

Please run on a linux based environment. We developed our code on Ubuntu. 

### Software Prerequisites

- CUDA
- zlib

### Clone the directory first
```
cd ~
git clone https://github.com/TheHarshShow/suffixArray
```
### Decompress data
We use the whole genome sequences of Drosophila ananassae, Drosophila yakuba, Drosophila melanogaster and Fugu as input datasets which were publicly available in UCSC Genome Browser. The input datasets are saved in the compressed format which needs to be decompressed before usage.

You might just want to decompress the dataset that you want to test on but the first dataset is the default so decompress that one at least.
```
cd suffixArray/data
xz --decompress droYak2_new_sanitised.fa.xz
xz --decompress dm62.fa.xz
xz --decompress Fugu2.fa.xz
xz --decompress GCA_018148915.1.2.fa.xz
```
## <a name="usage"></a> How to use

Once repository is cloned and input datasets are decompressed, run the code. 
```
cd ..
./run-commands.sh
```
If you're using DSMLP, use the ssh command from the assignments (replace the username and directory names appropriately)

```
ssh user@dsmlp-login.ucsd.edu /opt/launch-sh/bin/launch.sh -c 8 -g 1 -m 16 -i yatisht/ece284-wi23:latest -f ./suffixArray/run-commands.sh
```

The output looks like this. Here it prints the time taken by the GPU algorithm (Parallel Time), the O(nlogn) CPU algorithm (Sequential Time), the speedup and whether the two algorithms computed the same array or not.
```
Sequence name: CHR4
Sequence size: 1000000
Parallel Time: 56647595 nanoseconds
Sequential Time: 1298890504 nanoseconds
Speed up: 22.9293
Both CPU and GPU algorithms computed the same suffix array.
```

The default parameters (sequence length and dataset) are in the run-commands.sh file. Feel free to change them upto 10^8 bps.
```
# From the run-commands.sh file:
./suffixArray 1000000 ../data/droYak2_new_sanitised.fa
```
## <a name="references"></a> References

[1] P. Flick and S. Aluru, "Parallel distributed memory construction of suffix and longest common prefix arrays," SC '15: Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, Austin, TX, USA, 2015, pp. 1-10, doi: 10.1145/2807591.2807609.

