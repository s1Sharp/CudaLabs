#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include <stdio.h>

#define N 1000


__device__ bool Check(const float& x, const float& y)
{
    return x * x + y * y <= 1;
}

__global__ void piCalc(float* count) {

    float x = float(blockIdx.x) / N;
    float y = float(threadIdx.x) / N;
    Check(x, y) ? count[threadIdx.x * N + blockIdx.x] = 1 : count[threadIdx.x * N + blockIdx.x] = 0;
}

int main()
{
    float* sum = new float[N * N];
    float* count;

    cudaMalloc((void**)&count, N * N * sizeof(float));

    piCalc << < N, N >> > (count);

    cudaMemcpy(sum, count, N * N * sizeof(float), cudaMemcpyDeviceToHost);


    float ans = 0;
    for (int i = 0; i < N * N; ++i) {
        ans += sum[i];
    }
    printf("Pi is %f\n", ans * 4 / N / N);

    delete [] sum;
    cudaFree(count);
    return 0;
}