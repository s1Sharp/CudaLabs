
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "meraVector.h"
#include  "simpson.h"

const size_t vectorSize = 100;

int main()
{
    meraVector(vectorSize);
    linspace(0.f, 1.f, 5);
    linspace(0.f, 1.f, 5, false);

    integralSimpson(1.f, 2.f, 512);
    return 0;
}

