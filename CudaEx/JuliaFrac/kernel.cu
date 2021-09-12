#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../../common/book.h"
#include "../../common/cpu_bitmap.h"
#include "cuComplex.h"
#include <iostream>
#include <memory>

#define DIM 1000
#pragma comment (lib, "../../lib/glut64.lib")

using std::cout;


struct ptrDeleter {
    template <typename T>
    void operator()(T * ptr) { delete ptr; cout << "\nDeleted\n"; }
};

template <typename T>
using unique_p = std::unique_ptr <T, ptrDeleter>;


int julia(int x, int y)
{
    static const float scale = 1.5;
    float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
    float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

    cuComplex c(-0.8, 0.156);
    cuComplex z(jx, jy);

    unique_p <cuComplex> uptr (new cuComplex(-0.8f, 0.156f));

    for (int i = 0; i < 200; i++) {
        z = z * z + c;
        if (z.magnitude2() > 1000)
            return 0;
    }
    return 1;
}

void kernel(unsigned char* ptr)
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
//Zn+1 = Zn^2 + C

int main()
{
    CPUBitmap bitmap(DIM, DIM);
    unsigned char* ptr = bitmap.get_ptr();
    kernel(ptr);
    cout << "suq";
}
