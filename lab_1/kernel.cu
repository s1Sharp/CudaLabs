#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

void print_info();

__global__ void add(int* a, int* b, int* c) 
{
    *c = *a + *b;
}

int main() 
{
    int a, b, c;
    // host copies of variables a, b & c
    int* d_a, * d_b, * d_c;
    // device copies of variables a, b & c
    int size = sizeof(int);
    // Allocate space for device copies of a, b, c
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);
    // Setup input values  
    c = int();
    a = 777u;
    b = 333u;
    // Copy inputs to device
    cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);
    // Launch add() kernel on GPU
    add << <1, 1 >> > (d_a, d_b, d_c);
    // Copy result back to host
    cudaError err = cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) 
    {
        printf("CUDA error copying to Host: %s\n", cudaGetErrorString(err));
    }
    printf("result sum of %d and %d is %d\n",a , b, c);
    print_info();
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}

void print_info()
{
    int            deviceCount;
    cudaDeviceProp devProp;

    cudaGetDeviceCount(&deviceCount);

    printf("Found %d devices\n", deviceCount);

    for (int device = 0; device < deviceCount; device++)
    {
        cudaGetDeviceProperties(&devProp, device);

        printf("Device %d\n", device);
        printf("Compute capability     : %d.%d\n", devProp.major, devProp.minor);
        printf("Name                   : %s\n", devProp.name);
        printf("Total Global Memory    : %d\n", devProp.totalGlobalMem);
        printf("Shared memory per block: %d\n", devProp.sharedMemPerBlock);
        printf("Registers per block    : %d\n", devProp.regsPerBlock);
        printf("Warp size              : %d\n", devProp.warpSize);
        printf("Max threads per block  : %d\n", devProp.maxThreadsPerBlock);
        printf("Total constant memory  : %d\n", devProp.totalConstMem);
        printf("Total mmultiProcessor Count  : %d\n", devProp.multiProcessorCount);
    };

    return;
}
