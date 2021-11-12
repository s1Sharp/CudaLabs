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

template<typename T>
inline T sum_sq(T x)
{
    return (x * (x + 1) * (2 * x + 1) / 6);
}

const int N = 64 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

template <typename T>
__global__ void dot(T* a, T* b, T* c)
{
    __shared__ T cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    T temp = 0;
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


template <typename T>
__host__ void kernel()
{
    T* a, * b, c, * partial_c;
    T* dev_a, * dev_b, * dev_partial_c;

    // initialize clocks 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    //mem on cpu
    a = (T*)malloc(N * sizeof(T));
    b = (T*)malloc(N * sizeof(T));
    partial_c = (T*)malloc(blocksPerGrid * sizeof(T));

    cudaMalloc((void**)&dev_a,
        N * sizeof(T));
    cudaMalloc((void**)&dev_b,
        N * sizeof(T));
    cudaMalloc((void**)&dev_partial_c,
        N * sizeof(T));


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
    cudaMemcpy(dev_a, a, N * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(T), cudaMemcpyHostToDevice);

    //kernel
    cudaEventRecord(start, 0);
    dot << < blocksPerGrid, threadsPerBlock >> > (dev_a, dev_b, dev_partial_c);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float   elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("\n\nelapsed time %f ms by %s\n\n", elapsedTime, typeid(T).name());
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    //copy massiv back
    cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(T), cudaMemcpyDeviceToHost);
    //end solve on cpu

    c = 0;
    for (int i = 0; i < blocksPerGrid; i++)
    {
        c += partial_c[i];
    }
    printf("value gpu %.4g = %.4g?\n", c, 2 * sum_sq((float)(N - 1)));
}

int main()
{
    kernel<float>();
    kernel<double>();
    return 0;
}