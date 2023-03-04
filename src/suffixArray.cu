#include "suffixArray.cuh"
#include <iostream>
#include <thrust/sort.h>
#include <thrust/device_vector.h>


__global__ void assignIndexes(size_t l, char* seq, uint32_t* indexes){
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int bs = blockDim.x;
    int gs = gridDim.x;


    for(int i = bx*bs+tx; i < l; i+=bs*gs){
        indexes[i] = i;
    }

    if(bx == 0 && tx==0){
        seq[l-1] = '$';
    }

}

__global__ void rebucket(size_t l, char* seq, uint32_t* bucket2){
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int bs = blockDim.x;
    int gs = gridDim.x;

    __shared__ uint32_t workbench[2*3000];

    int out=0,in=1;
    // printf("Change!!!\n");

    for(int i = bx*bs+tx; i < l; i+=bs*gs){

        if(i > 0 && seq[i] != seq[i-1]){
            workbench[out*(l+1) + i] = i;
        } else {
            workbench[out*(l+1) + i] = 0;
        }
    }
    __syncthreads();

    if(bx == 0){
        for(int offset = 1; offset < l; offset <<= 1){
            //swap buffers
            out = 1 - out;
            in = 1 - in;
            if(tx <= l){
                workbench[out*(l+1)+tx] = (tx>=offset) ? max(workbench[in*(l+1)+tx], workbench[in*(l+1)+tx-offset]) : workbench[in*(l+1)+tx];
            }
            __syncthreads();
        }
    }
    __syncthreads();

    for(int i = bx*bs+tx; i < l; i+=bs*gs){
        bucket2[i] = workbench[out*(l+1)+i];
    }

}

__global__ void SAToISA(size_t l, uint32_t* bucket2, uint32_t* indexes, uint32_t* bucket){
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int bs = blockDim.x;
    int gs = gridDim.x;
    for(int i = bs*bx+tx; i < l; i+=bs*gs){
        bucket[indexes[i]] = bucket2[i];
    }
}

__global__ void shift(size_t l, uint32_t bucket){

}

void SuffixArray::Sequence::allocateSequenceArray(size_t n){
    l = n;
    cudaMalloc(&sequence, l*sizeof(char));
    cudaMalloc(&indexes, l*sizeof(uint32_t));
    cudaMalloc(&bucket, l*sizeof(uint32_t));
    cudaMalloc(&bucket2, l*sizeof(uint32_t));
}

void SuffixArray::Sequence::copyToGPU(char* cpuSequence){
    cudaMemcpy(sequence, cpuSequence, (l-1)*sizeof(char), cudaMemcpyHostToDevice);
}

void SuffixArray::Sequence::createSuffixArray(){
    int numBlocks = 1024; // i.e. number of thread blocks on the GPU
    int blockSize = 512; 

    assignIndexes<<<numBlocks, blockSize>>>(l, sequence, indexes);
    thrust::device_ptr<uint32_t> indexesPtr(indexes);
    thrust::device_ptr<char> sequencePtr(sequence);
    thrust::sort_by_key(sequencePtr,sequencePtr+l,indexesPtr);

    rebucket<<<1, 1024>>>(l, sequence, bucket2);

    SAToISA<<<numBlocks,blockSize>>>(l,bucket2,indexes,bucket);

}

#define HANDLE_GPU_ERROR(ans)		\
{									\
	cudaError_t errorNum = ans;		\
	if (errorNum != cudaSuccess)	\
	{								\
		std::cout 	<< std::dec <<	cudaGetErrorString( errorNum ) << " in " << __FILE__ << " at line " << __LINE__ << std::endl; \
		exit(-1);					\
	}								\
}

void SuffixArray::Sequence::copyToCPU(uint32_t* cpuIndexes, char* seq){
    // printf("L:::: %d\n", l);
    // for(size_t i = 0; i < 20; i++){
    //     // std::cout << cpuIndexes[i] << std::endl;
    //     printf("%d\n", cpuIndexes[i]);
    // }

    // int err = cudaMemcpy(cpuIndexes, indexes, l*sizeof(uint32_t), cudaMemcpyDeviceToHost);
    // if (err != cudaSuccess) {
    //     fprintf(stderr, "GPU_ERROR: cudaMemCpy failed! %d\n", err);
    //     exit(1);
    // }

    HANDLE_GPU_ERROR( cudaMemcpy(cpuIndexes, bucket, l*sizeof(uint32_t), cudaMemcpyDeviceToHost) );
    HANDLE_GPU_ERROR( cudaMemcpy(seq, sequence, l*sizeof(char), cudaMemcpyDeviceToHost) );
    // for(int i = 0; i < 20; i++){
    //     std::cout << cpuIndexes[i] << std::endl;
    // }

    // cudaMemcpy(cpuIndexes, indexes, l*sizeof(uint32_t), cudaMemcpyDeviceToHost);
    // cudaMemcpy(cpuIndexes, indexes, (l) * sizeof(uint32_t), cudaMemcpyDeviceToHost);
}

void SuffixArray::Sequence::freeSequenceArray(){
    cudaFree(sequence);
    cudaFree(indexes);
}