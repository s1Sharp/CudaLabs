
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

// 1gb = 1073741824 bytes; float = 4 bytes; => size = 268435456 size_t 
const size_t size = 1024 * 1024 * 256;

const size_t count = 25;

int main()
{
    // initialize clocks 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    float* dev_tmp;
    float* tmp = (float*)malloc( size* sizeof(float));
    cudaMalloc((void**)&dev_tmp,size * sizeof(float));


    cudaEventRecord(start, 0);

    for (size_t iter = 0; iter < count; iter++) {
        cudaMemcpy(dev_tmp, tmp, size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(tmp, dev_tmp, size * sizeof(float), cudaMemcpyDeviceToHost);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float   elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    float speed = 2.f * 1000 * 25 / elapsedTime; // GB per sec
    std::cout <<  speed << " GB/sec";


    cudaFree(dev_tmp);
    free(tmp);

    return 0;
}

