#pragma once

namespace{
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



int* linspace(int, int, size_t) = delete;
int* linspace(int, int, size_t, bool) = delete;

template<typename T = float>
T* linspace(T start,T  stop, size_t num, bool endpoint = true)
{
	T* ptr = nullptr;
	if (start > stop || num <= 0)
		return nullptr;
	
	
	T step = (stop - start) / (num - static_cast<int>(endpoint));
	if (!step)
	{
		return &start;
	}

	ptr = new T[num];
	
	for (size_t i = 0; i < num; i++)
	{
		ptr[i] = start + step * i;
	}

	return ptr;
}

template<typename T = float>
__device__ T f(const T x)
{
	return x * x + 1 / (3 * x + 5);
}

template<typename T = float>
__global__ void simpson(T start,T stop,size_t num, T* result, bool endpoint = true )
{
	extern __shared__ T cache[];
	const T step = (stop - start) / (num - static_cast<int>(endpoint));

	size_t cacheIndex = threadIdx.x;
	cache[cacheIndex] = 0;
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < num)
	{
	const double x1 = start + step * tid;
	const double x2 = start + step * (tid+1);
	cache[cacheIndex] = (x2 - x1) / 6.0 * (f(x1) + 4.0 * f(0.5 * (x1 + x2)) + f(x2));
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


void integralSimpson(const int& start, const int& stop, size_t N) = delete;

template<typename T = float>
void integralSimpson(const T& start, const T& stop, size_t N)
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

	simpson << <BLOCKS_PER_GRID, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(float) >> > (start,stop, N ,dev_ptr);

	cudaMemcpy(&ptr, dev_ptr, sizeof(float) , cudaMemcpyDeviceToHost);

	
	printf("%f - result on gpu\n", ptr);

	cudaFree(dev_ptr);

	return;
}