#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include "suffixArray.cuh"
#include "kseq.h"
#include "zlib.h"

// For reading in the FASTA file
KSEQ_INIT2(, gzFile, gzread)

using namespace std;

// Sequential algorithm to compare with. Source: https://cp-algorithms.com/string/suffix-array.html#on2-log-n-approach
std::vector<int> sort_cyclic_shifts(string const& s) {
    int n = s.size();
    const int alphabet = 256;
    std::vector<int> p(n), c(n), cnt(max(alphabet, n), 0);
    for (int i = 0; i < n; i++)
        cnt[s[i]]++;
    for (int i = 1; i < alphabet; i++)
        cnt[i] += cnt[i-1];
    for (int i = 0; i < n; i++)
        p[--cnt[s[i]]] = i;
    c[p[0]] = 0;
    int classes = 1;
    for (int i = 1; i < n; i++) {
        if (s[p[i]] != s[p[i-1]])
            classes++;
        c[p[i]] = classes - 1;
    }
    std::vector<int> pn(n), cn(n);
    for (int h = 0; (1 << h) < n; ++h) {
        for (int i = 0; i < n; i++) {
            pn[i] = p[i] - (1 << h);
            if (pn[i] < 0)
                pn[i] += n;
        }
        fill(cnt.begin(), cnt.begin() + classes, 0);
        for (int i = 0; i < n; i++)
            cnt[c[pn[i]]]++;
        for (int i = 1; i < classes; i++)
            cnt[i] += cnt[i-1];
        for (int i = n-1; i >= 0; i--)
            p[--cnt[c[pn[i]]]] = pn[i];
        cn[p[0]] = 0;
        classes = 1;
        for (int i = 1; i < n; i++) {
            pair<int, int> cur = {c[p[i]], c[(p[i] + (1 << h)) % n]};
            pair<int, int> prev = {c[p[i-1]], c[(p[i-1] + (1 << h)) % n]};
            if (cur != prev)
                ++classes;
            cn[p[i]] = classes - 1;
        }
        c.swap(cn);
    }
    return p;
}

int main(int argc, char* argv[]){
    
    uint32_t SEQLEN = atoi(argv[1]);

    std::string refFilename = string(argv[2]);
    // std::string refFilename = "../data/droYak2_new_sanitised.fa";

    // Read file from dataset
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
    printf("Sequence size: %u\n", SEQLEN);

    // Creating string for sequential code
    std::string s;
    for(uint32_t i = 0; i < SEQLEN; i++)s+=record->seq.s[i];
    s+='$';

    uint32_t* cpuIndexes = new uint32_t[SEQLEN+1];

    // Cuda data structure
    SuffixArray::Sequence seq;

    // Allocate memory to arrays
    seq.allocateSequenceArray(SEQLEN+1);

    // Copy sequence to GPU
    seq.copyToGPU(record->seq.s);

    auto parStart = std::chrono::high_resolution_clock::now();

    // Compute Suffix Array on GPU
    seq.computeSuffixArray();

    auto parEnd = std::chrono::high_resolution_clock::now();
    std::chrono::nanoseconds parTime = parEnd - parStart;

    std::cout << "Parallel Time: " << parTime.count() << " nanoseconds" << std::endl;

    // Get suffix array from GPU
    seq.copyToCPU(cpuIndexes, record->seq.s);

    auto seqStart = std::chrono::high_resolution_clock::now();

    // Compute suffix array on CPU
    std::vector<int> v = sort_cyclic_shifts(s);

    auto seqEnd = std::chrono::high_resolution_clock::now();
    std::chrono::nanoseconds seqTime = seqEnd - seqStart;

    std::cout << "Sequential Time: " << seqTime.count() << " nanoseconds" << std::endl;
    std::cout << "Speed up: " << seqTime.count()*1.0/parTime.count() << std::endl;

    bool matched = true;

    // Compare CPU array to GPU array
    for(size_t i = 0; i < SEQLEN+1; i++){
        if(cpuIndexes[i] != v[i]){
            matched = false;
            break;
        }
    }

    if(matched){
        std::cout << "Both CPU and GPU algorithms computed the same suffix array." << std::endl;
    } else {
        std::cout << "Error: CPU and GPU algorithms computed different suffix arrays!" << std::endl;
    }

    seq.freeSequenceArray();

    delete[] cpuIndexes;
    
}