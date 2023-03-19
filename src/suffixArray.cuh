#include <stdint.h>
#include <stdio.h>

// CUDA suffix array data structure
namespace SuffixArray{

    struct Sequence {
        // comparing each value with previous value to check for singletons
        bool* singletonValues;
        // Sequence stored as array of characters
        char* sequence;
        // This is where the suffix array will be stored.
        uint32_t* indexes;
        // B1 and B2 from the algorithm
        uint32_t* bucket2;
        uint32_t* bucket;
        // To zip sort B1 and B2
        uint64_t* combinedBuckets;
        // Length of the sequence including the $ character
        int l;
        
        // Allocate arrays of size n
        void allocateSequenceArray(size_t n); 

        // Copy sequence to GPU
        void copyToGPU(char* cpuSequence);

        // Copy suffix array to CPU
        void copyToCPU(uint32_t* cpuIndexes, char* seq);

        // Compute the suffix array
        void computeSuffixArray();

        // Free GPU memory up at the end
        void freeSequenceArray();
    };
};