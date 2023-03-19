# Parallelizing Suffix Array Construction

Prerequisites: CUDA, zlib

Clone the directory first
```
cd ~
git clone https://github.com/TheHarshShow/suffixArray
```

Decompress data. You might just want to decompress the dataset that you want to test on but the first dataset is the default so decompress that one at least.
```
cd suffixArray/data
xz --decompress droYak2_new_sanitised.fa.xz
xz --decompress dm62.fa.xz
xz --decompress Fugu2.fa.xz
xz --decompress GCA_018148915.1.2.fa.xz
```

Then run the code. If you're using DSMLP, use the ssh command from the assignments.
```
cd ..
./run-commands.sh
```

The output looks like this. Here you can see the time taken by the GPU algorithm, the O(nlogn) CPU algorithm, the speedup and whether the two algorithms computed the same array or not.
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
