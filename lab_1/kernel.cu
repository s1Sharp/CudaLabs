#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <thread>
#include <functional>

void print_info(FILE *);

__global__ void add(int* a, int* b, int* c) 
{
    *c = *a + *b;
}

FILE* file = fopen("output.txt", "w");

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
    a = 7777u;
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

   
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}
