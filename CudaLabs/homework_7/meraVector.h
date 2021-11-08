#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <random>
#include <stdio.h>


template <typename T>
__global__ void kernel(T* ptr, size_t size)
{
    extern __shared__ T cache[];
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    size_t cacheIndex = threadIdx.x;

    cache[cacheIndex] = ptr[tid] * ptr[tid];

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
    ptr[blockIdx.x] = cache[0];
    return;
}

template <typename T>
void generateVector(T* _ptr, size_t size)
{
    srand(4721u);
    for (size_t i = 0; i < size; i++)
    {
        _ptr[i] = static_cast<T>(2);//(rand() % 2);
    }
    return;

}

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
    float* ptr = new float[N];
    generateVector(ptr, N);
    float checkResultCPU = checkResult(ptr, N);

    float* dev_ptr;
    cudaMalloc((void**)&dev_ptr, sizeof(float) * N);
    cudaMemcpy(dev_ptr, ptr, sizeof(float) * N, cudaMemcpyHostToDevice);

    size_t THREADS_PER_BLOCK = 64;
    size_t BLOCKS_PER_GRID =
        (N / THREADS_PER_BLOCK) + 1;

    kernel << <BLOCKS_PER_GRID, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(float) >> > (dev_ptr, N);

    cudaMemcpy(ptr, dev_ptr, sizeof(float) * N, cudaMemcpyDeviceToHost);

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