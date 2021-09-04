#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <ostream>

void print_info(FILE *);

__global__ void add(int* a, int* b, int* c) 
{
    *c = *a + *b;
}

int main() 
{
    FILE* file = fopen("output.txt", "w");

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
        fprintf(file,"CUDA error copying to Host: %s\n", cudaGetErrorString(err));
    }
    fprintf(file,"result sum of %d and %d is %d\n",a , b, c);
    print_info(file);
    fclose(file);
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}

void print_info(FILE * f)
{
    int            deviceCount;
    cudaDeviceProp devProp;

    cudaGetDeviceCount(&deviceCount);

    fprintf(f,"Found %d devices\n", deviceCount);

    for (int device = 0; device < deviceCount; device++)
    {
        cudaGetDeviceProperties(&devProp, device);

        fprintf(f,"Device %d\n", device);
        fprintf(f,"Compute capability     : %d.%d\n", devProp.major, devProp.minor);
        fprintf(f,"Name                   : %s\n", devProp.name);
        fprintf(f,"Total Global Memory    : %llu\n", devProp.totalGlobalMem);
        fprintf(f,"Shared memory per block: %d\n", devProp.sharedMemPerBlock);
        fprintf(f,"Registers per block    : %d\n", devProp.regsPerBlock);
        fprintf(f,"Warp size              : %d\n", devProp.warpSize);
        fprintf(f,"Max threads per block  : %d\n", devProp.maxThreadsPerBlock);
        fprintf(f,"Total constant memory  : %d\n", devProp.totalConstMem);
        fprintf(f,"Total mmultiProcessor Count  : %d\n", devProp.multiProcessorCount);
    };

    return;
}
