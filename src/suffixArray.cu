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

    // Set last character of sequence array
    if(bx==0 && tx==0){
        seq[l-1] = '$';
    }

}

__global__ void rebucket(size_t l, char* sequence, uint32_t* bucket2){
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int bs = blockDim.x;
    int gs = gridDim.x;

    __shared__ uint32_t workbench[3000];

    for(int i = bx*bs+tx; i < l; i+=bs*gs){
        if(i > 0 && sequence[i] != sequence[i-1]){
            workbench[i] = i;
        } else {
            workbench[i] = 0;
        }
    }
    __syncthreads();

    int pout=0, pin=1;

    if(bx == 0){
        for(int offset = 1; offset < l; offset *= 2){
            if(tx <= l){
                pout = 1 - pout;
                pin = 1 - pin;
                if(tx >= offset){
                    workbench[pout*(l+1)+tx] = max(workbench[pin*(l+1)+tx], workbench[pin*(l+1)+tx-offset]);
                } else {
                    workbench[pout*(l+1)+tx] = workbench[pin*(l+1)+tx];
                }
            }
            __syncthreads();
        }

        for(int i = tx; i < l; i += bs){
            bucket2[i] = workbench[pout*(l+1)+i];
        }
    }
}

__global__ void rebucket(size_t l, uint64_t* combinedBuckets, uint32_t* bucket2){
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int bs = blockDim.x;
    int gs = gridDim.x;

    __shared__ uint32_t workbench[3000];

    for(int i = bx*bs+tx; i < l; i+=bs*gs){
        if(i > 0 && combinedBuckets[i] != combinedBuckets[i-1]){
            workbench[i] = i;
        } else {
            workbench[i] = 0;
        }
    }
    __syncthreads();

    int pout=0, pin=1;

    if(bx == 0){
        for(int offset = 1; offset < l; offset *= 2){
            if(tx <= l){
                pout = 1 - pout;
                pin = 1 - pin;
                if(tx >= offset){
                    workbench[pout*(l+1)+tx] = max(workbench[pin*(l+1)+tx], workbench[pin*(l+1)+tx-offset]);
                } else {
                    workbench[pout*(l+1)+tx] = workbench[pin*(l+1)+tx];
                }
            }
            __syncthreads();
        }

        for(int i = tx; i < l; i += bs){
            bucket2[i] = workbench[pout*(l+1)+i];
        }
    }
}

__global__ void shift(size_t l, uint32_t* bucket, uint32_t* bucket2, uint64_t* combinedBuckets, size_t offset){
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int bs = blockDim.x;
    int gs = gridDim.x;

    for(int i = bs*bx+tx; i < l - offset; i+=bs*gs){
        bucket2[i] = bucket[i+offset];
        combinedBuckets[i] = ((uint64_t)bucket[i] << 32) + bucket2[i];
    }
    for(int i = l - offset + bs*bx+tx; i < l; i+=bs*gs){
        bucket2[i] = 0;
        combinedBuckets[i] = ((uint64_t)bucket[i] << 32) + bucket2[i];
    }
}

__global__ void SAToISA(size_t l, uint32_t* indexes, uint32_t* bucket2, uint32_t* bucket){
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int bs = blockDim.x;
    int gs = gridDim.x;

    for(int i = bx*bs+tx; i < l; i+=bs*gs){
        bucket[indexes[i]] = bucket2[i];
    }
}

__device__ bool d_allSingletonAnswer;

__global__ void allSingleton(size_t l, uint32_t* bucket){
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int bs = blockDim.x;
    int gs = gridDim.x;

    __shared__ bool orArray[3000];
    for(int i = bx*bs+tx; i < l-1; i+=bs*gs){
        orArray[i] = (bucket[i] == bucket[i+1]);
    }
    __syncthreads();

    if(bx == 0){
        for(uint32_t s = l/2; s > 0; s>>=1){
            if(tx < s && tx+s < l-1){
                orArray[tx] |= orArray[tx+s];
            }
            __syncthreads();
        }

        d_allSingletonAnswer = !orArray[0];
        // printf("All Singleton: %d\n", d_allSingletonAnswer);
        // return orArray[0];
    }
}

void SuffixArray::Sequence::allocateSequenceArray(size_t n){
    l = n;
    cudaMalloc(&sequence, l*sizeof(char));

    cudaMalloc(&indexes, l*sizeof(uint32_t));
    cudaMalloc(&bucket2, l*sizeof(uint32_t));
    cudaMalloc(&bucket, l*sizeof(uint32_t));
    cudaMalloc(&combinedBuckets, l*sizeof(uint64_t));
}

void SuffixArray::Sequence::copyToGPU(char* cpuSequence){
    cudaMemcpy(sequence, cpuSequence, (l-1)*sizeof(char), cudaMemcpyHostToDevice);
}

void SuffixArray::Sequence::computeSuffixArray(){
    int numBlocks = 1024; // i.e. number of thread blocks on the GPU
    int blockSize = 512; 

    assignIndexes<<<numBlocks, blockSize>>>(l, sequence, indexes);
    thrust::device_ptr< uint32_t > indexesPtr(indexes);
    thrust::device_ptr< char > sequencePtr(sequence);
    thrust::device_ptr< uint64_t > combinedBucketsPtr(combinedBuckets);

    thrust::sort_by_key(sequencePtr, sequencePtr + l, indexesPtr);

    size_t offset = 1;
    bool allSingletonAnswer = false;
    
    rebucket<<<1, min(1024, l)>>>(l,sequence,bucket2);

    while(!allSingletonAnswer){
        SAToISA<<<numBlocks, blockSize>>>(l, indexes, bucket2, bucket);
        shift<<<numBlocks,blockSize>>>(l,bucket,bucket2,combinedBuckets,offset);

        assignIndexes<<<numBlocks, blockSize>>>(l, sequence, indexes);

        thrust::sort_by_key(combinedBucketsPtr, combinedBucketsPtr + l, indexesPtr);
        rebucket<<<1, min(1024, l)>>>(l,combinedBuckets,bucket2);

        allSingleton<<<numBlocks, blockSize>>>(l,bucket2);  
        cudaMemcpyFromSymbol(&allSingletonAnswer, d_allSingletonAnswer, sizeof(allSingletonAnswer), 0, cudaMemcpyDeviceToHost);
        
        std::cout << allSingletonAnswer << std::endl;

        offset<<=1;
    }
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

    HANDLE_GPU_ERROR( cudaMemcpy(cpuIndexes, indexes, l*sizeof(uint32_t), cudaMemcpyDeviceToHost) );
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