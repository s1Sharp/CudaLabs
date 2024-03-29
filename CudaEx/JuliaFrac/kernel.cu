﻿#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../../common/book.h"
#include "../../common/cpu_bitmap.h"
#include "cuComplex.h"
#include <iostream>
#include <memory>
#include "timer.h"
#include "bmout.h"



#define DIM 1000
#pragma comment (lib, "../../lib/glut64.lib")

static Timer t;
using std::cout;

using defer = std::shared_ptr<void>;

struct ptrDeleter {
    template <typename T>
    void operator()(T * ptr) { cudaFree(ptr); /* cout << "\nDeleted\n"; */ }
};

template <typename T>
using unique_p = std::unique_ptr <T, ptrDeleter>;


int julia(int x, int y)
{
    static const float scale = 1.5;
    float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
    float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

    Complex c(-0.8, 0.156);
    Complex z(jx, jy);

    //unique_p <cuComplex> uptr (new cuComplex(-0.8f, 0.156f));

    for (int i = 0; i < 200; i++) {
        z = z * z + c;
        if (z.magnitude2() > 1000)
            return 0;
    }
    return 1;
}

void kernelCPU(unsigned char* ptr)
{
    for (int y = 0; y < DIM; y++) 
    {
        for (int x = 0; x < DIM; x++)
        {
            int offset = x + y * DIM;

            int juliaVal = julia(x, y);
            ptr[offset * 4 + 0] = 255 * juliaVal;
            ptr[offset * 4 + 1] = 0;
            ptr[offset * 4 + 2] = 0;
            ptr[offset * 4 + 3] = 255;
        }
    }

}

__device__ int julia_dev(int x, int y)
{
    static const float scale = 1.5;
    float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
    float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

    cuComplex c(-0.8, 0.156);
    cuComplex z(jx, jy);

    for (int i = 0; i < 200; i++) {
        z = z * z + c;
        if (z.magnitude2() > 1000)
            return 0;
    }
    return 1;
}
__global__ void kernelGPU(unsigned char* ptr)
{
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;

    int juliaVal = julia_dev(x, y);
    ptr[offset * 4 + 0] = 255 * juliaVal;
    ptr[offset * 4 + 1] = 0;
    ptr[offset * 4 + 2] = 0;
    ptr[offset * 4 + 3] = 255;
}
//Zn+1 = Zn^2 + C

int main()
{
    CPUBitmap bitmap(DIM, DIM);
    unsigned char* ptr = bitmap.get_ptr();
    unsigned char* dev_ptr;

    HANDLE_ERROR(cudaMalloc((void**)&dev_ptr, bitmap.image_size()));
    //defer for free cuda mem 
    defer d(nullptr, [dev_ptr](...) 
        { cudaFree(dev_ptr);  cout << "free"; });

    dim3 grid(DIM, DIM);

    
    t.start();
    kernelGPU <<< grid, 1 >>> (dev_ptr);
    //kernelCPU(ptr);
    

    HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(),
        dev_ptr,
        bitmap.image_size(),
        cudaMemcpyDeviceToHost));

    //
    cout << t.elapsedSeconds();
    t.stop();
    SaveImage("output.bmp", (const char*)bitmap.get_ptr(),DIM,DIM);
    bitmap.display_and_exit();
    cudaFree(dev_ptr);
    cout<<"suq";
    
}
