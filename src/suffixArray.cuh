#include <stdint.h>
#include <stdio.h>

namespace SuffixArray{

    struct Sequence {
        bool* singletonValues;
        char* sequence;
        uint32_t* indexes;
        uint32_t* bucket2;
        uint32_t* bucket;
        uint64_t* combinedBuckets;
        
        int l;
        void allocateSequenceArray(size_t n); 
        void copyToGPU(char* cpuSequence);
        
        void copyToCPU(uint32_t* cpuIndexes, char* seq);

        void computeSuffixArray();

        void freeSequenceArray();
    };
};