#pragma once

template<typename T = float>
__global__ void rectangle(T start, T stop, size_t num, T* result, bool endpoint = true)
{
	extern __shared__ T cache[];
	const T step = (stop - start) / (num - static_cast<int>(endpoint));

	size_t cacheIndex = threadIdx.x;
	cache[cacheIndex] = 0;
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < num)
	{
		const double x1 = start + step * tid;
		const double x2 = start + step * (tid + 1);
		cache[cacheIndex] = (x2 - x1) * f((x1 + x2) / 2);
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
		atomicAdd(result, cache[0]);
	return;
}


int integralRectangle(const int& start, const int& stop, size_t N) = delete;

template<typename T = float>
T integralRectangle(const T& start, const T& stop, size_t N)
{
	if (N < 1)
		N = defaultNUM;
	float ptr = 0;

	N = NextOrEqualPower2(N);

	float* dev_ptr;
	cudaMalloc((void**)&dev_ptr, sizeof(float));

	size_t THREADS_PER_BLOCK = std::min(
		std::max(64, static_cast<int>(N)),
		1024);
	size_t BLOCKS_PER_GRID =
		(N / THREADS_PER_BLOCK) + 1;

	rectangle << <BLOCKS_PER_GRID, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(float) >> > (start, stop, N, dev_ptr);

	cudaMemcpy(&ptr, dev_ptr, sizeof(float), cudaMemcpyDeviceToHost);


	printf("%f - result on gpu (rectangle)\n", ptr);

	cudaFree(dev_ptr);

	return ptr;
}