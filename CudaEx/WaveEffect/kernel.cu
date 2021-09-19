
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../../common/book.h"
#include "../../common/cpu_anim.h"
#include <iostream>
#include <memory>
#include "timer.h"

#define DIM1 1920
#define DIM2 1080

struct DataBlock {
    unsigned char* dev_bitmap;
    CPUAnimBitmap* bitmap;
};

void cleanup(DataBlock* d) {
    cudaFree(d->dev_bitmap);
    printf("free ptr\n");
}

__global__ void kernel(unsigned char* ptr, int ticks)
{
    //отображаем пару threadIdx/BlockIdx на позицию пикселя
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    //вычисление в этой позиции

    float fx = x - DIM1 / 2;
    float fy = y - DIM2 / 2;
    float d = sqrtf(fx * fx + fy * fy);
    unsigned char grey = (unsigned char)(128.0f + 127.0f * 
                                        (cos(d / 10.0f - ticks / 7.0f) /   
                                        (d / 10.0f + 1.0f)));


    ptr[offset * 4 + 0] = grey;
    ptr[offset * 4 + 1] = grey;
    ptr[offset * 4 + 2] = grey;
    ptr[offset * 4 + 3] = 255;
    }

void generate_frame(DataBlock* d, int ticks)
{
    dim3 blocks(DIM1 / 16, DIM2 / 16);
    dim3 threads(16, 16);
    kernel << <blocks, threads >> > (d->dev_bitmap, ticks);

    HANDLE_ERROR(cudaMemcpy(d->bitmap->get_ptr(),
                            d->dev_bitmap,
                            d->bitmap->image_size(),
                            cudaMemcpyDeviceToHost));

}

int main()
{
    DataBlock data;
    CPUAnimBitmap bitmap(DIM1, DIM2, &data);
    data.bitmap = &bitmap;

    HANDLE_ERROR(cudaMalloc((void**)&data.dev_bitmap, bitmap.image_size()));

    bitmap.anim_and_exit(   (void(*) (void*, int))generate_frame, (void(*)(void*))cleanup);

    return 0;
}
