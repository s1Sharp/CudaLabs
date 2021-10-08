#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"

#include <memory>
#include <ctime>
#include <stdio.h>
#include <chrono>
#include <algorithm>


using defer = std::shared_ptr<void>;

#define WIDTH 32
#define HEIGHT 32
#define hBLOCKS 6
#define wBLOCKS 6

//return resurl of this expression (x*x + y*y <= 1)
__device__ bool inCircle(curandState_t* state)
{
    float x = curand_uniform(state);
    float y = curand_uniform(state);
    return x * x + y * y <= 1.0f;
}


__global__ void CalculatePointsIntheCircle(int* result, int randseed)
{
    curandState_t state;
    (threadIdx.x + blockDim.x * blockIdx.x)* threadIdx.y + blockDim.y * blockIdx.y;
    unsigned long long seed = (threadIdx.x + blockDim.x * blockIdx.x) + (threadIdx.y + blockDim.y * blockIdx.y) * (randseed % 1000);

    //init curand
    curand_init(seed, 0, 0, &state);

    if (inCircle(&state))
    {
        atomicAdd(result, 1);
    }
}

int main()
{
    int count = 0;
    int* dev_count;

    cudaMalloc((void**)&dev_count, sizeof(int));
    
    // starting the timer here so that we include the cost of
    // all of the operations on the GPU.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    //use un_ptr, that don`t forget free memory
    defer _(nullptr, [&](...)
        { cudaFree(dev_count);  cudaEventDestroy(start); cudaEventDestroy(stop);  printf("free"); });

    dim3 blocks(hBLOCKS, wBLOCKS, 1);
    dim3 threads(HEIGHT, WIDTH, 1);

    int randseed = std::chrono::duration_cast<std::chrono::milliseconds>
        (std::chrono::system_clock::now().time_since_epoch()).count();

    CalculatePointsIntheCircle <<<blocks, threads >>> (dev_count, randseed);

    cudaMemcpy(&count, dev_count, sizeof(int), cudaMemcpyDeviceToHost);
    // result pi
    float pi = (4.0f * static_cast<float>(count)) / static_cast<float>(HEIGHT * WIDTH * hBLOCKS * wBLOCKS);
    printf("Result pi %f \n", pi);

    //print elapsed time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float   elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf( "Elapsed time %3.1f ms\n", elapsedTime );

    return 0;
}