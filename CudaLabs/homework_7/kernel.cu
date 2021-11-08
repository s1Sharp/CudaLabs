
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "meraVector.h"

const size_t vectorSize = 100;

int main()
{
    meraVector(vectorSize);

    return 0;
}

