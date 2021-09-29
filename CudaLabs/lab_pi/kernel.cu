#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"

#include <memory>
#include <ctime>
#include <stdio.h>
#include <chrono>
#include <algorithm>

#define ACCURACY 12828879
#define NUM_OF_ITER 10


using defer = std::shared_ptr<void>;

//cuRandom API prototype 
//                   __device__ float curand_uniform(curandState_t* state) ->
//single normally distributed float with mean 0.0 and standard deviation 1.0

__global__ void Pi(int* count, curandState_t* globalState,unsigned long long seed,int accuracy)
{
	int indx = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	curand_init(seed, indx, 0, &globalState[indx]);

	curandState_t localState = globalState[indx];
	while (indx < accuracy) 
	{
		float temp_x = curand_uniform(&localState);
		float temp_y = curand_uniform(&localState);
		float z = temp_x * temp_x + temp_y * temp_y;
		if (z < 1.0f) {
			atomicAdd(count, 1);
		}
		indx += stride;
	}
}

int main()
{
	int count;
	int* dev_count;
	float res = 0.0f;
	float* sort_res = new float[NUM_OF_ITER];

	unsigned long long seed = time(NULL);

	cudaDeviceProp  prop;
	cudaGetDeviceProperties(&prop, 0);
	
	//max Speed with blocks * 2
	int blocks = prop.multiProcessorCount *  2;
	int threads = prop.maxThreadsPerBlock;
	int total = (ACCURACY - (ACCURACY % threads)); //last indx of __global__ func
	printf("kernel start with %u blocks and %u threads, total %u\n", blocks, threads,total);

	//for random eq
	curandState_t* devState;
	cudaMalloc((void**)&devState, total * sizeof(curandState_t));

	//allocate memory on GPU
	cudaMalloc((void**)&dev_count, sizeof(int));

	// starting the timer here so that we include the cost of
	// all of the operations on the GPU.
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);


	//use un_ptr, that don`t forget free memory
	defer _(nullptr, [dev_count, devState, start, stop,sort_res ](...)
		{ cudaFree(dev_count); cudaFree(devState); cudaEventDestroy(start); cudaEventDestroy(stop); delete[] sort_res; printf("free"); });


	for (int iter = 0; iter < NUM_OF_ITER; iter++)
	{
		//kernel
		Pi << < blocks, threads >> > (dev_count, devState, seed, total);

		//copy result to HOST
		cudaMemcpy(&count, dev_count, sizeof(int), cudaMemcpyDeviceToHost);

		//upd seed
		seed = std::chrono::duration_cast<std::chrono::milliseconds>
															(std::chrono::system_clock::now().time_since_epoch()).count() + iter;
		cudaMemset(dev_count, 0, sizeof(int));

		float tempres = count * 4.0f / total;

		printf("res of %u iter PI: %f \n", iter, tempres);
		sort_res[iter] = tempres;
		res += tempres;
	}

	std::sort(sort_res, sort_res + 10);
	printf("______sorted result_____ \n");
	for (size_t i = 0; i < 10; i++)
	{
		printf("\tPI: %f \n",*(sort_res + i));
	}

	//print elapsed time
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float   elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Time to generate:  %3.1f ms in mean foreach iter, total time: %3.1f ms \n", elapsedTime/NUM_OF_ITER, elapsedTime);
	
	//result
	printf("result = %f\n", res / NUM_OF_ITER);

	return 0;
}
