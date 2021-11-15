
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include "device_functions.h"
#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <vector>


namespace {
	template <typename T>
	std::vector<T>& randVector(size_t size)
	{
		using namespace std;
		random_device rnd_device;
		// Specify the engine and distribution.
		mt19937 mersenne_engine{ rnd_device() };  // Generates random integers
		uniform_int_distribution<int> dist{ 1, 52 };
		auto gen = [&dist, &mersenne_engine]() {
			return dist(mersenne_engine);
		};
		std::vector<T>* v = new std::vector<T> (size);
		generate(v->begin(), v->end(), gen);
		// Optional
		return *v;

	}

	const size_t defaultNUM = 64;
}

void check_cuda_error(const char* message)
{
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("ERROR: %s: %s\n", message,
			cudaGetErrorString(err));
}

template <typename T>
__global__ void kernel(T* res, T s, size_t num)
{
	extern __shared__ T cache[];

	size_t cacheIndex = threadIdx.x;
	cache[cacheIndex] = 0;
	int tid = 1 + threadIdx.x + blockIdx.x * blockDim.x ;
	if (tid < num)
	{
		
		cache[cacheIndex] = (float)1 / powf((float)tid, s); //вычисление очередного слагаемого
	}

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
	if (cacheIndex == i)
		atomicAdd(res, cache[0]);
	return;
}



int main(int argc, char** argv)
{
	size_t N = 1024 *1024;
	auto v = randVector<float>(N);

	float* res_d; // Результаты наустройстве
	float res = 0;
	float stepen = 2.0;
	cudaMalloc((void**)&res_d, sizeof(float));
	check_cuda_error("Allocating memory on GPU");
	cudaMemcpy(res_d, &res, sizeof(float), cudaMemcpyHostToDevice);
	check_cuda_error("Allocating memory on GPU");
	// Рамеры грида и блока на GPU
	size_t THREADS_PER_BLOCK = std::min(std::max(64, static_cast<int>(N)), 1024);
	size_t BLOCKS_PER_GRID = (N / THREADS_PER_BLOCK) + 1;
	kernel << <BLOCKS_PER_GRID, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(float) >> > (res_d, stepen , N);

	cudaThreadSynchronize();
	check_cuda_error("Executing kernel");

	cudaMemcpy(&res, res_d, sizeof(float), cudaMemcpyDeviceToHost);
	// Копируем результаты на хост
	check_cuda_error("Copying results from GPU");
	cudaFree(res_d);
	check_cuda_error("Freeing device memory");

	printf("Dzeta(%.1f) = %.12f\n",stepen, res);
	printf("Theoretical value of Dzeta(2.0)  = pi*pi/6 = 1,64493406685\n");
	return 0;
}
