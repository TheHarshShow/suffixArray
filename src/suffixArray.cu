#include "suffixArray.cuh"
#include <iostream>

__global__ void assignIndexes(size_t l, char* seq, uint32_t* indexes){
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int bs = blockDim.x;
    int gs = gridDim.x;

    for(int i = bx*bs+tx; i < l; i+=bs*gs){
        indexes[i] = i;
    }
    // __syncthreads();

    // if(bx == 0 && tx == 0){
    //     for(int i = 0; i < 20; i++){
    //         printf("%d %c\n", indexes[i], seq[i]);
    //     }
    // }

}

void SuffixArray::Sequence::allocateSequenceArray(size_t n){
    l = n;
    cudaMalloc(&sequence, l*sizeof(char));
    cudaMalloc(&indexes, l*sizeof(uint32_t));
}

void SuffixArray::Sequence::copyToGPU(char* cpuSequence){
    cudaMemcpy(sequence, cpuSequence, l*sizeof(char), cudaMemcpyHostToDevice);

    int numBlocks = 1024; // i.e. number of thread blocks on the GPU
    int blockSize = 512; 

    assignIndexes<<<numBlocks, blockSize>>>(l, sequence, indexes);
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

void SuffixArray::Sequence::copyToCPU(uint32_t* cpuIndexes){
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

    HANDLE_GPU_ERROR( cudaMemcpy(cpuIndexes, indexes, l*sizeof(uint32_t), cudaMemcpyDeviceToHost) );
    for(int i = 0; i < 20; i++){
        std::cout << cpuIndexes[i] << std::endl;
    }

    // cudaMemcpy(cpuIndexes, indexes, l*sizeof(uint32_t), cudaMemcpyDeviceToHost);
    // cudaMemcpy(cpuIndexes, indexes, (l) * sizeof(uint32_t), cudaMemcpyDeviceToHost);
}

void SuffixArray::Sequence::freeSequenceArray(){
    cudaFree(sequence);
    cudaFree(indexes);
}