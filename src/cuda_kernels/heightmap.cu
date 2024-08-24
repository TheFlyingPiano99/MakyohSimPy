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

    double height = sin(2.0 * M_PI_d * pos.x) * sin(4.0 * M_PI_d * pos.y);
    //double height = pos.x;

    // End of function
    unsigned int idx = get_array_index_2d();
    heightmap[idx] = height;
}
