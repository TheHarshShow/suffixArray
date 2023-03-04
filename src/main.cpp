#include <iostream>
#include <fstream>
#include <stdlib.h>
#include "suffixArray.cuh"
#include "kseq.h"
#include "zlib.h"

// For reading in the FASTA file
KSEQ_INIT2(, gzFile, gzread)

int main(int argc, char* argv[]){
    std::cout << "Hello World!" << std::endl;
    
    std::string refFilename = "../data/Mississippi.fa";

    gzFile fp = gzopen(refFilename.c_str(), "r");
    if (!fp) {
        fprintf(stderr, "ERROR: Cannot open file: %s\n", refFilename.c_str());
        exit(1);
    }
    kseq_t *record = kseq_init(fp);
    int n;
    if ((n = kseq_read(record)) < 0) {
        fprintf(stderr, "ERROR: No reference sequence found!\n");
        exit(1);
    }
    printf("Sequence name: %s\n", record->name.s);
    printf("Sequence size: %zu\n", record->seq.l);

    // uint32_t* cpuIndexes = (uint32_t *)malloc(sizeof(uint32_t)*(record->seq.l+1));
    uint32_t* cpuIndexes = new uint32_t[record->seq.l+1];

    SuffixArray::Sequence seq;
    seq.allocateSequenceArray(record->seq.l+1);

    seq.copyToGPU(record->seq.s);
    seq.computeSuffixArray();

    seq.copyToCPU(cpuIndexes, record->seq.s);

    for(size_t i = 0; i < 20; i++){
        std::cout << cpuIndexes[i] << " " << record->seq.s[i] << std::endl;
    }

    seq.freeSequenceArray();

    delete[] cpuIndexes;
    // free(cpuIndexes);

    std::cout << "Hello World 5!" << std::endl;
    
}