#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <vector>
#include <algorithm>
// #include <pair>
#include "suffixArray.cuh"
#include "kseq.h"
#include "zlib.h"

// For reading in the FASTA file
KSEQ_INIT2(, gzFile, gzread)

int main(int argc, char* argv[]){
    std::cout << "Hello World!" << std::endl;
    
    std::string refFilename = "../data/HIVSequence.fa";

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

    std::string s;
    for(int i = 0; i < record->seq.l; i++)s+=record->seq.s[i];
    s+='$';
    std::vector< std::pair< std::string, int > > v({{s, 0}});
    for(int i = 1; i < s.length(); i++){
        std::rotate(s.begin(), s.begin()+1, s.end());
        v.push_back({s, i});
    }
    std::sort(v.begin(), v.end());

    // uint32_t* cpuIndexes = (uint32_t *)malloc(sizeof(uint32_t)*(record->seq.l+1));
    uint32_t* cpuIndexes = new uint32_t[record->seq.l+1];

    SuffixArray::Sequence seq;
    // seq.allocateSequenceArray(record->seq.l+1);

    seq.allocateSequenceArray(record->seq.l+1);

    seq.copyToGPU(record->seq.s);
    seq.computeSuffixArray();

    seq.copyToCPU(cpuIndexes, record->seq.s);

    for(size_t i = 0; i < record->seq.l+1; i++){
        if(cpuIndexes[i] != v[i].second){
            std::cout << "CHANGE!!! " << cpuIndexes[i] << " " << v[i].second << std::endl;
        }
    }

    seq.freeSequenceArray();

    delete[] cpuIndexes;

    // std::cout << "Hello World 5!" << std::endl;
    
}