
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

#define uint16_t unsigned short


namespace {
	const size_t defaultNUM = 64;
}

uint16_t NextOrEqualPower2(size_t N)
{
	uint16_t size = static_cast<uint16_t>(N);
	uint16_t mask = 0x8000; //(16 byte, 32768, 1000 0000 0000 0000)

	//calculates the nearest 2^n from above
	while (!(mask & size))
		mask >>= 1;
	return  static_cast<size_t> (mask << (mask != size));
}

// Проверка на ошибку выполнения функций из cuda

void check_cuda_error(const char* message)
{
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("ERROR: %s: %s\n", message,
			cudaGetErrorString(err));
}

template <typename T>
__global__ void kernel(T* res,size_t num)
{
	extern __shared__ T cache[];
	const T step = 1 / (float)(num);

	size_t cacheIndex = threadIdx.x;
	cache[cacheIndex] = 0;
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < num)
	{
		T x0 = step * tid;
		T x1 = step * (tid + 1);
		T y0 = sqrtf(1 - x0 * x0);
		T y1 = sqrtf(1 - x1 * x1);

		cache[cacheIndex] = (y0 + y1) * step / 2.f; // Площадьтрапеции
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
	size_t N = 128;
	if (N < 1)
		N = defaultNUM;
	N = NextOrEqualPower2(N);

	float* res_d; // Результаты наустройстве
	float res = 0;
	cudaMalloc((void**)&res_d, sizeof(float)); 
	check_cuda_error("Allocating memory on GPU");

	// Рамеры грида и блока на GPU
	size_t THREADS_PER_BLOCK = std::min(std::max(64, static_cast<int>(N)), 1024);
	size_t BLOCKS_PER_GRID = ( N / THREADS_PER_BLOCK) + 1;
	kernel <<<BLOCKS_PER_GRID, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(float) >>> (res_d,N);

	cudaThreadSynchronize(); 
	check_cuda_error("Executing kernel");

	cudaMemcpy(&res, res_d, sizeof(float), cudaMemcpyDeviceToHost);
	// Копируем результаты на хост
	check_cuda_error("Copying results from GPU");
	cudaFree(res_d);
	check_cuda_error("Freeing device memory");
	
	res *= 4;
	printf("PI = %.12f\n", res);
	return 0;
}
