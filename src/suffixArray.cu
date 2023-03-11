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

    for(int i = bx*bs+tx; i < l; i+=bs*gs){
        if(i > 0 && sequence[i] != sequence[i-1]){
            bucket2[i] = i;
        } else {
            bucket2[i] = 0;
        }
    }
}

__global__ void prefixScanKernel(size_t l, uint32_t* prefixArray){
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int bs = blockDim.x;
    int gs = gridDim.x;

    __shared__ uint32_t workbench[3000];

    for(size_t i = bx*bs; i < l; i+=bs*gs){
        int si = min(l - i, (size_t)bs);
        for(size_t j = i+tx; j < i+si; j+=bs){
            workbench[j-i] = prefixArray[j];
        }
        __syncthreads();
        // for(size_t j = 1; j < si; j++){
        //     workbench[j] = max(workbench[j-1],workbench[j]);
        // }
        // __syncthreads();
        // for(size_t j = i+1; j < i+si; j++){
        //     prefixArray[j] = max(prefixArray[j-1],prefixArray[j]);
        // }

        int pout=0, pin=1;

        for(int offset = 1; offset < si; offset *= 2){
            if(tx <= si){
                pout = 1 - pout;
                pin = 1 - pin;
                if(tx >= offset){
                    workbench[pout*(si+1)+tx] = max(workbench[pin*(si+1)+tx], workbench[pin*(si+1)+tx-offset]);
                } else {
                    workbench[pout*(si+1)+tx] = workbench[pin*(si+1)+tx];
                }
            }
            __syncthreads();
        }

        for(int j = tx; j < si; j += bs){
            prefixArray[i+j] = workbench[pout*(si+1)+j];
        }

    }
}

__global__ void copyMaxes(size_t l, size_t newL, uint32_t* prefixArray, uint32_t* newBucket){
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int bs = blockDim.x;
    int gs = gridDim.x;

    if(tx == 0){
        for(size_t i = bx*bs; i < l - bs; i+=bs*gs){
            newBucket[i/bs] = prefixArray[i+bs-1];
        }
    }
}

__global__ void propogate(size_t l, size_t newL, uint32_t* prefixArray, uint32_t* newBucket){
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int bs = blockDim.x;
    int gs = gridDim.x;

    for(size_t i = bx*bs; i < l - bs; i+=bs*gs){
        uint32_t newMax = newBucket[i/bs];
        prefixArray[i+bs+tx] = max(prefixArray[i+bs+tx], newMax);
    }
}

void prefixScan(size_t l, uint32_t* bucket2, size_t numBlocks, size_t blockSize){
    // run prefix kernel
    prefixScanKernel<<<numBlocks,blockSize>>>(l,bucket2);

    if(l > blockSize){
        size_t newL = (l-1)/blockSize;
        // create GPU array
        uint32_t* newBucket;
        cudaMalloc(&newBucket, newL*sizeof(uint32_t));
        // copy to newBucket
        copyMaxes<<<numBlocks,blockSize>>>(l,newL,bucket2,newBucket);

        prefixScan(newL, newBucket, numBlocks, blockSize);
        //propogate kernel
        propogate<<<numBlocks,blockSize>>>(l,newL,bucket2,newBucket);

        cudaFree(newBucket);
    }
}

// __global__ void rebucket(size_t l, char* sequence, uint32_t* bucket2){
//     int tx = threadIdx.x;
//     int bx = blockIdx.x;
//     int bs = blockDim.x;
//     int gs = gridDim.x;

//     __shared__ uint32_t workbench[3000];

//     for(int i = bx*bs+tx; i < l; i+=bs*gs){
//         if(i > 0 && sequence[i] != sequence[i-1]){
//             workbench[i] = i;
//         } else {
//             workbench[i] = 0;
//         }
//     }
//     __syncthreads();

//     int pout=0, pin=1;

//     if(bx == 0){
//         for(int offset = 1; offset < l; offset *= 2){
//             if(tx <= l){
//                 pout = 1 - pout;
//                 pin = 1 - pin;
//                 if(tx >= offset){
//                     workbench[pout*(l+1)+tx] = max(workbench[pin*(l+1)+tx], workbench[pin*(l+1)+tx-offset]);
//                 } else {
//                     workbench[pout*(l+1)+tx] = workbench[pin*(l+1)+tx];
//                 }
//             }
//             __syncthreads();
//         }

//         for(int i = tx; i < l; i += bs){
//             bucket2[i] = workbench[pout*(l+1)+i];
//         }
//     }
// }

__global__ void rebucket(size_t l, uint64_t* combinedBuckets, uint32_t* bucket2){
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int bs = blockDim.x;
    int gs = gridDim.x;

    for(int i = bx*bs+tx; i < l; i+=bs*gs){
        if(i > 0 && combinedBuckets[i] != combinedBuckets[i-1]){
            bucket2[i] = i;
        } else {
            bucket2[i] = 0;
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

__device__ bool d_allSingletonAnswer = 1;

__global__ void allSingleton(size_t l, uint32_t* bucket){
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int bs = blockDim.x;
    int gs = gridDim.x;

    // if(bx == 0 && tx == 0){
    //     d_allSingletonAnswer = 1;
    // }
    // bool allSingleton = true;

    // for(int i = bs*bx+tx+1; i < l; i+=bs*gs){
    //     if(bucket[i] == bucket[i-1]){
    //         allSingleton = false;
    //     }
    // }

    if(bx == 0 && tx == 0){
        d_allSingletonAnswer = 1;
        for(int i = 1; i < l; i++){
            if(bucket[i] == bucket[i-1]){
                d_allSingletonAnswer = 0;
            }
        }
    }

    // __shared__ bool orArray[3000];
    // for(int i = bx*bs+tx; i < l-1; i+=bs*gs){
    //     orArray[i] = (bucket[i] == bucket[i+1]);
    // }
    // __syncthreads();

    // if(bx == 0){
    //     for(uint32_t s = l/2; s > 0; s>>=1){
    //         if(tx < s && tx+s < l-1){
    //             orArray[tx] |= orArray[tx+s];
    //         }
    //         __syncthreads();
    //     }

    //     d_allSingletonAnswer = !orArray[0];
    //     // printf("All Singleton: %d\n", d_allSingletonAnswer);
    //     // return orArray[0];
    // }
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
    int blockSize = 1024; 

    assignIndexes<<<numBlocks, blockSize>>>(l, sequence, indexes);
    thrust::device_ptr< uint32_t > indexesPtr(indexes);
    thrust::device_ptr< char > sequencePtr(sequence);
    thrust::device_ptr< uint64_t > combinedBucketsPtr(combinedBuckets);

    thrust::sort_by_key(sequencePtr, sequencePtr + l, indexesPtr);

    size_t offset = 1;
    bool allSingletonAnswer = false;
    
    rebucket<<<numBlocks, blockSize>>>(l,sequence,bucket2);
    prefixScan(l,bucket2,numBlocks,blockSize);

    while(!allSingletonAnswer){
        SAToISA<<<numBlocks, blockSize>>>(l, indexes, bucket2, bucket);
        shift<<<numBlocks,blockSize>>>(l,bucket,bucket2,combinedBuckets,offset);

        assignIndexes<<<numBlocks, blockSize>>>(l, sequence, indexes);

        thrust::sort_by_key(combinedBucketsPtr, combinedBucketsPtr + l, indexesPtr);
        rebucket<<<numBlocks, blockSize>>>(l,combinedBuckets,bucket2);
        prefixScan(l,bucket2,numBlocks,blockSize);

        allSingleton<<<numBlocks, blockSize>>>(l,bucket2);  
        cudaMemcpyFromSymbol(&allSingletonAnswer, d_allSingletonAnswer, sizeof(allSingletonAnswer), 0, cudaMemcpyDeviceToHost);
        
        uint32_t* cpuIndexes2 = new uint32_t[l];
        // cudaMemcpy(cpuIndexes2, indexes, l*sizeof(uint32_t), cudaMemcpyDeviceToHost);
        // for(int i = 0; i < l; i++){
        //     std::cout << cpuIndexes2[i] << " ";
        // }
        // std::cout << std::endl;

        // std::cout << allSingletonAnswer << " " << offset << std::endl;

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