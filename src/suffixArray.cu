#include "suffixArray.cuh"
#include <iostream>
#include <thrust/sort.h>
#include <thrust/device_vector.h>


// Assign integer indexes to the characters in the sequence
__global__ void assignIndexes(size_t l, char* seq, uint32_t* indexes){
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int bs = blockDim.x;
    int gs = gridDim.x;

    // index of ith character is i
    for(int i = bx*bs+tx; i < l; i+=bs*gs){
        indexes[i] = i;
    }

    // Set last character of sequence array
    if(bx==0 && tx==0){
        seq[l-1] = '$';
    }

}

// Rebucketing kernel WITHOUT the suffix array computation. There are two rebucketing kernels. This one handles char input.
__global__ void rebucket(size_t l, char* sequence, uint32_t* bucket2){
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int bs = blockDim.x;
    int gs = gridDim.x;

    // wherever we see change, assign that index to i, otherwise 0
    for(int i = bx*bs+tx; i < l; i+=bs*gs){
        if(i > 0 && sequence[i] != sequence[i-1]){
            bucket2[i] = i;
        } else {
            bucket2[i] = 0;
        }
    }
}


// Prefix max scan for just one recursive call.
__global__ void prefixScanKernel(size_t l, uint32_t* prefixArray){
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int bs = blockDim.x;
    int gs = gridDim.x;

    __shared__ uint32_t workbench[3000];

    // iterate through chunks in sequence. ith block handles ith chunk
    for(size_t i = bx*bs; i < l; i+=bs*gs){
        int si = min(l - i, (size_t)bs);

        // copy to shared memory
        for(size_t j = i+tx; j < i+si; j+=bs){
            workbench[j-i] = prefixArray[j];
        }
        __syncthreads();

        int pout=0, pin=1;

        // prefix scan over a block
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


        // copy to global memory
        for(int j = tx; j < si; j += bs){
            prefixArray[i+j] = workbench[pout*(si+1)+j];
        }

    }
}

// Copy the max of each segment to a new array. This kernel is used in prefix max computation
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

// Propogate the max until the previous segment (computed after recursively performing prefix scan on maxes) into the current segment
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

// Function that computes prefix scan. Calls the kernels perfexScanKernel, copyMaxes and propogate
void prefixScan(size_t l, uint32_t* bucket2, size_t numBlocks, size_t blockSize){
    // run prefix kernel
    prefixScanKernel<<<numBlocks,blockSize>>>(l,bucket2);

    if(l > blockSize){
        size_t newL = (l-1)/blockSize;
        // create GPU array
        uint32_t* newBucket;
        cudaMalloc(&newBucket, newL*sizeof(uint32_t));
        
        // copy maxes of each segment to newBucket
        copyMaxes<<<numBlocks,blockSize>>>(l,newL,bucket2,newBucket);

        // recursively compute prefix scan
        prefixScan(newL, newBucket, numBlocks, blockSize);

        // propogate kernel to propogate maxes forward
        propogate<<<numBlocks,blockSize>>>(l,newL,bucket2,newBucket);

        cudaFree(newBucket);
    }
}

// Create a boolean array where the ith element is compared with the i-1th element to see if they are different. Used in all singleton computation
__global__ void computeAllSingletonArray(size_t l, uint32_t* bucket, bool* singletonValues){
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int bs = blockDim.x;
    int gs = gridDim.x;

    for(int i = bs*bx+tx+1; i < l; i+=bs*gs){
        singletonValues[i-1] = (bucket[i] != bucket[i-1]);
    }
}

// Used to check if every element in the array is unique. Done through parallel reduction on the AND operator
__global__ void allSingletonKernel(size_t l, size_t newL, bool* bucket, bool* newBucket){
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int bs = blockDim.x;
    int gs = gridDim.x;

    __shared__ bool workbench[1024];
    // Since this kernel is called recursively on a reduced array, it needs to compute that reduced array. In case of final recursive step, the size of the reduced array is 1
    for(int i = bs*bx+tx; i < newL; i+=bs*gs){
        newBucket[i] = true;
    }

    // Compute the AND for each block and put that into newBucket on which AND reduction will be done again
    for(int i = bs*bx; i < l; i+=bs*gs){
        if(i+tx < l){
            // copy to shared memory
            workbench[tx] = bucket[i+tx];
            int si = min((size_t)bs, l-i);
            __syncthreads();

            // compute parallel reduction
            for(uint32_t s = (si+1)/2; s > 0; s>>=1){
                if(tx < s && tx+s < si){
                    workbench[tx] = (workbench[tx+s] && workbench[tx]);
                }
                __syncthreads();
            }
            
            // store the AND value to the reduced array. Again, if the original array was smaller than a block, the reduced array will only have one element
            if(tx == 0){
                newBucket[i/bs]=workbench[0];
            }
        }
    }

}


// Function that computes if every element in a sorted array is unique
bool allSingleton(size_t l, bool* bucket, size_t numBlocks, size_t blockSize){

    // Size of reduced array after AND of each block is computed
    size_t newL = (l+blockSize-1)/blockSize;
    bool allSingletonValue = true;

    // Array where max of each bucket will be stored
    bool* newBucket;
    cudaMalloc(&newBucket, newL*sizeof(bool));

    // Run all singleton kernel
    allSingletonKernel<<<numBlocks,blockSize>>>(l,newL,bucket,newBucket);

    // If there was more than one block's data in the original array, recursively compute allSingleton
    if(newL > 1){
        allSingletonValue = allSingleton(newL, newBucket, numBlocks, blockSize);
        cudaFree(newBucket);
        return allSingletonValue;
    }
    
    // If the array was small (less than a block), just return its all singleton value
    cudaMemcpy(&allSingletonValue, newBucket, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(newBucket);

    return allSingletonValue;

}

// rebucket if input was an integer array. Doesn't include reduction.
__global__ void rebucket(size_t l, uint64_t* combinedBuckets, uint32_t* bucket2){
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int bs = blockDim.x;
    int gs = gridDim.x;

    // If different, store i. Otherwise 0
    for(int i = bx*bs+tx; i < l; i+=bs*gs){
        if(i > 0 && combinedBuckets[i] != combinedBuckets[i-1]){
            bucket2[i] = i;
        } else {
            bucket2[i] = 0;
        }
    }
}

// Shift the whole array by offset
__global__ void shift(size_t l, uint32_t* bucket, uint32_t* bucket2, uint64_t* combinedBuckets, size_t offset){
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int bs = blockDim.x;
    int gs = gridDim.x;

    for(int i = bs*bx+tx; i < l - offset; i+=bs*gs){
        bucket2[i] = bucket[i+offset];
        combinedBuckets[i] = ((uint64_t)bucket[i] << 32) + bucket2[i];
    }

    // Store zeroes at final positions
    for(int i = l - offset + bs*bx+tx; i < l; i+=bs*gs){
        bucket2[i] = 0;
        combinedBuckets[i] = ((uint64_t)bucket[i] << 32) + bucket2[i];
    }
}

// SA to ISA permutation. Takes maximum time due to non-coalesced memory access pattern
__global__ void SAToISA(size_t l, uint32_t* indexes, uint32_t* bucket2, uint32_t* bucket){
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int bs = blockDim.x;
    int gs = gridDim.x;

    for(int i = bx*bs+tx; i < l; i+=bs*gs){
        bucket[indexes[i]] = bucket2[i];
    }
}

// Allocate memory to arrays
void SuffixArray::Sequence::allocateSequenceArray(size_t n){
    l = n;
    cudaMalloc(&sequence, l*sizeof(char));
    cudaMalloc(&singletonValues, (l-1)*sizeof(char));
    cudaMalloc(&indexes, l*sizeof(uint32_t));
    cudaMalloc(&bucket2, l*sizeof(uint32_t));
    cudaMalloc(&bucket, l*sizeof(uint32_t));
    cudaMalloc(&combinedBuckets, l*sizeof(uint64_t));
}

// Copy sequence from CPU to GPU
void SuffixArray::Sequence::copyToGPU(char* cpuSequence){
    cudaMemcpy(sequence, cpuSequence, (l-1)*sizeof(char), cudaMemcpyHostToDevice);
}

// The main function that computes the suffix array
void SuffixArray::Sequence::computeSuffixArray(){
    int numBlocks = 1024; // i.e. number of thread blocks on the GPU
    int blockSize = 1024; 

    // Assign index i to i^th character
    assignIndexes<<<numBlocks, blockSize>>>(l, sequence, indexes);
    thrust::device_ptr< uint32_t > indexesPtr(indexes);
    thrust::device_ptr< char > sequencePtr(sequence);
    thrust::device_ptr< uint64_t > combinedBucketsPtr(combinedBuckets);

    // Pair sort the sequence with indexes
    thrust::sort_by_key(sequencePtr, sequencePtr + l, indexesPtr);

    size_t offset = 1;
    bool allSingletonAnswer = false;
    
    // First rebucketing step
    rebucket<<<numBlocks, blockSize>>>(l,sequence,bucket2);
    prefixScan(l,bucket2,numBlocks,blockSize);

    while(!allSingletonAnswer){
        SAToISA<<<numBlocks, blockSize>>>(l, indexes, bucket2, bucket);
        shift<<<numBlocks,blockSize>>>(l,bucket,bucket2,combinedBuckets,offset);

        assignIndexes<<<numBlocks, blockSize>>>(l, sequence, indexes);

        // combined sort of b1 and b2
        thrust::sort_by_key(combinedBucketsPtr, combinedBucketsPtr + l, indexesPtr);

        // Rebucketing again
        rebucket<<<numBlocks, blockSize>>>(l,combinedBuckets,bucket2);
        prefixScan(l,bucket2,numBlocks,blockSize);

        // Compute boolean of whether the ith element is equal to the i-1th element or not
        computeAllSingletonArray<<<numBlocks, blockSize>>>(l,bucket2,singletonValues);
        
        // check if any of the booleans are false
        allSingletonAnswer = allSingleton(l-1,singletonValues,numBlocks,blockSize);

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

// copy to CPU
void SuffixArray::Sequence::copyToCPU(uint32_t* cpuIndexes, char* seq){

    HANDLE_GPU_ERROR( cudaMemcpy(cpuIndexes, indexes, l*sizeof(uint32_t), cudaMemcpyDeviceToHost) );
    HANDLE_GPU_ERROR( cudaMemcpy(seq, sequence, l*sizeof(char), cudaMemcpyDeviceToHost) );
    
}

void SuffixArray::Sequence::freeSequenceArray(){
    cudaFree(sequence);
    cudaFree(indexes);
    cudaFree(singletonValues);
    cudaFree(bucket2);
    cudaFree(bucket);
    cudaFree(combinedBuckets);
}