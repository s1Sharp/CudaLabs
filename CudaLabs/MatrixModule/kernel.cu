
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "matrix.h"

template <typename T>
void kernel(mymatrix::MATRIX<T> m1, mymatrix::MATRIX<T> m2)
{
    m1 += m2;
};

int main()
{
    using namespace mymatrix;
    mymatrix::test();


    return 0;
}

