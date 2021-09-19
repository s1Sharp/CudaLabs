#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../../common/book.h"
#include <iostream>
#include <memory>
#include "timer.h"



#define N 3 * 1024

static Timer t;
using std::cout;

using defer = std::shared_ptr<void>;

struct ptrDeleter {
    template <typename T>
    void operator()(T* ptr) { cudaFree(ptr); /* cout << "\nDeleted\n"; */ }
};

template <typename T>
using unique_p = std::unique_ptr <T, ptrDeleter>;

__global__ void add(int* a,int *b,int *c)
{
    int thid = threadIdx.x + blockIdx.x * blockDim.x;
    while (thid < N) {
        c[thid] = a[thid] + b[thid];
        thid += blockDim.x * gridDim.x;
    }

}
//Zn+1 = Zn^2 + C

int main()
{
    int a[N], b[N], c[N];
    int* dev_a, * dev_b, *dev_c;

    t.start();

    HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(int)));

    defer d(nullptr, [dev_a,dev_b,dev_c](...)
        {   cudaFree(dev_a);  cout << "free a\n";
            cudaFree(dev_b);  cout << "free b\n";
            cudaFree(dev_c);  cout << "free c\n"; });

    for (int i = 0; i < N; i++) {
        a[i] = i-i/4+4;
        b[i] = i * i;
    }
    HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int),cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));
    
    add << <128, 128 >> > (dev_a, dev_b, dev_c);

    HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));
    
    t.stop();
    cout << t.elapsedSeconds()<< std::endl;

    bool suc = true;
    for (int i = 0; i < N; i++) {
        if ((a[i] + b[i]) != c[i])
        {
            printf("mistake: %d + %d != %d\n", a[i], b[i], c[i]);
            suc = false;
        }
    }
    if (suc) printf("i did it!\n");
    return 0;
}
