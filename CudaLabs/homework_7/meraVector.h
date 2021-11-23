#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <random>
#include <stdio.h>


template <typename T>
__global__ void kernel(T* ptr, size_t size)
{
    //cache for reduction size N * sizeof(float) bytes
    extern __shared__ T cache[];

    //index for vector valie
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    //index for reduction inside block 
    size_t cacheIndex = threadIdx.x;

    cache[cacheIndex] = ptr[tid] * ptr[tid];

    //sync before reduction
    __syncthreads();

    // reduction
    size_t i = blockDim.x / 2;

    while (i != 0) {
        if (cacheIndex < i)
        {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }
    //save result reduction in ptr[index of block]
    ptr[blockIdx.x] = cache[0];
    return;
}

//generate random vector
template <typename T>
void generateVector(T* _ptr, size_t size)
{
    srand(4721u);
    for (size_t i = 0; i < size; i++)
    {
        _ptr[i] = static_cast<T>(rand() % 999999);
    }
    return;

}

//func caculate evclid norma on CPU
template <typename T>
T checkResult(T* _ptr, size_t size)
{
    T result(0);
    for (size_t i = 0; i < size; i++)
    {
        result += _ptr[i] * _ptr[i];
    }

    return std::sqrt(result);
}



void meraVector(size_t N)
{
    //generate vector size N
    float* ptr = new float[N];
    generateVector(ptr, N);

    printf("\n");
    for (int i = 0; i < N; i++)
    {
        printf("%f ", ptr[i]);
    }
    printf("\n");

    //calculate result on CPU
    float checkResultCPU = checkResult(ptr, N);

    //cpy vector to device memory
    float* dev_ptr;
    cudaMalloc((void**)&dev_ptr, sizeof(float) * N);
    cudaMemcpy(dev_ptr, ptr, sizeof(float) * N, cudaMemcpyHostToDevice);


    //define threads and block dimension
    size_t THREADS_PER_BLOCK = 64;
    size_t BLOCKS_PER_GRID =
        (N / THREADS_PER_BLOCK) + 1;

    //start GPU kernel
    kernel << <BLOCKS_PER_GRID, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(float) >> > (dev_ptr, N);

    //cpy result to host memory
    cudaMemcpy(ptr, dev_ptr, sizeof(float) * N, cudaMemcpyDeviceToHost);


    //calculate reduction result on cpu
    float result = 0.f;
    for (size_t i = 0; i < BLOCKS_PER_GRID; i++)
    {
        result += ptr[i];
    }

    result = std::sqrt(result);
    printf("%f - result on gpu\n", result);
    printf("%f - check result on cpu\n", checkResultCPU);

    delete[]ptr;
    cudaFree(dev_ptr);

    return;
}