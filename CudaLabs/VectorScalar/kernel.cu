#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <memory>
#include <iostream>


#ifndef __CUDACC__
#define __CUDACC__
#include <device_functions.h>
#endif


template<typename T>
inline T imin(T a, T b)
{
    return (a < b ? a : b);
}

const int N = 32 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);


__global__ void dot(float* a, float* b, float* c)
{
    __shared__ float cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    float temp = 0;
    while (tid < N)
    {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    //save variable in cache
    cache[cacheIndex] = temp;

    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0)
    {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }
    if (cacheIndex == 0) 
        c[blockIdx.x] = cache[0];
}

using defer = std::shared_ptr<void>;


int main()
{
    // initialize clocks 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float* a, * b, c, * partial_c;
    float* dev_a, * dev_b, * dev_partial_c;

    //mem on cpu
    a = (float*)malloc(N * sizeof(float));
    b = (float*)malloc(N * sizeof(float));
    partial_c = (float*)malloc(blocksPerGrid * sizeof(float));

    cudaMalloc((void**)&dev_a,
        N * sizeof(float));
    cudaMalloc((void**)&dev_b,
        N * sizeof(float));
    cudaMalloc((void**)&dev_partial_c,
        N * sizeof(float));


    defer _1(nullptr, [a, b, partial_c, c](...)
        { delete[] a; delete[] b; delete[] partial_c; std::cout << "free host memory\n"; });
    defer _0(nullptr, [dev_a, dev_b, dev_partial_c](...)
        { cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_partial_c); std::cout << "free device memory\n"; });


    for (int i = 0; i < N; i++)
    {
        a[i] = i;
        b[i] = i * 2;
    }

    //mem cpy
    cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(start, 0);

    //kernel
    dot << < blocksPerGrid, threadsPerBlock >> > (dev_a, dev_b, dev_partial_c);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float   elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Elapsed time for kernel = %f sec\n", elapsedTime);
    //copy massiv back
    cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
    //end solve on cpu

    c = 0;
    for (int i = 0; i < blocksPerGrid; i++)
    {
        c += partial_c[i];
    }

    return 0;
}
