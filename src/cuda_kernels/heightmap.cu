#include "cuda_kernels/common.cu"


extern "C" __global__
void heightmap(
    double* __restrict__ heightmap,
    double delta_x,
    double delta_y
)
{
    uint2 pixel = get_voxel_coords_2d();
    // The position takes the (0, 0) value in the upper left corner of the mirror viewed from the canvas.
    double2 pos = double2{(double)pixel.x * delta_x, (double)pixel.y * delta_y};
    // Function:

    //double height = 0.001 * sin(2.0 * M_PI_d * pos.x) /** sin(4.0 * M_PI_d * pos.y)*/;
    //double height = 0.001 * sin(2.0 * M_PI_d * pos.x) /** sin(4.0 * M_PI_d * pos.y)*/;
    double height = 0.0;
    double x0 = 0.48;
    double x1 = 0.52;
    double r = 10.0;
    if (pos.x > x0 && pos.x < x1)
    {
        height = sqrt(r*r - pow((x1 - x0) / 2.0, 2)) - sqrt(r*r - pow(pos.x - (x0 + x1) / 2.0, 2));
    }

    // End of function
    unsigned int idx = get_array_index_2d();
    heightmap[idx] = height;
}
