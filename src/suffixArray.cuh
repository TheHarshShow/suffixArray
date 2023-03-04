#include <stdint.h>
#include <stdio.h>

namespace SuffixArray{

    struct Sequence {
        char* sequence;
        uint32_t* indexes;
        uint32_t* bucket2;
        uint32_t* bucket;
        int l;
        void allocateSequenceArray(size_t n); 
        void copyToGPU(char* cpuSequence);
        void createSuffixArray();
        void copyToCPU(uint32_t* cpuIndexes,  char* seq);

        void freeSequenceArray();
    };
};