
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "meraVector.h"
#include "simpson.h"
#include "rectangle.h"

const size_t vectorSize = 100;
const size_t vectorSize2 = 3;
const size_t vectorSize3 = 333;

int main()
{
    meraVector(vectorSize);
    meraVector(vectorSize2);
    meraVector(vectorSize3);
    /*
    linspace(0.f, 1.f, 5);
    linspace(0.f, 1.f, 5, false);

    integralSimpson(1.f, 2.f, 512);

    integralRectangle(1.f, 2.f, 512);
    */


    return 0;
}

