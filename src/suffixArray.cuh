#include <stdint.h>
#include <stdio.h>

namespace SuffixArray{

    struct Sequence {
        char* sequence;
        uint32_t* indexes;
        int l;
        void allocateSequenceArray(size_t n); 
        void copyToGPU(char* cpuSequence);
        
        void copyToCPU(uint32_t* cpuIndexes);

        void freeSequenceArray();
    };
};