#include "cuda_kernels/common.cu"


extern "C" __global__
void canvas(
    double* __restrict__ heightmap,
    double* canvas,
    double delta_x,
    double delta_y,
    double distance // Mirror distance from the canvas
)
{
    uint2 mirror_pixel = get_voxel_coords_2d();
    // The position takes the (0, 0) value in the upper left corner of the mirror viewed from the canvas.
    double2 mirror_pos = double2{(double)mirror_pixel.x * delta_x, (double)mirror_pixel.y * delta_y};
    unsigned int idx = get_array_index_2d();
    double height = heightmap[idx];

    // Calculate reflected light and canvas position:
    // Use mirror_pos
    double reflection = height; // Test
    double2 canvas_pos = mirror_pos; // Test

    // End of reflection and canvas pos calculation

    uint2 canvas_pixel = uint2{(unsigned int)(canvas_pos.x / delta_x), (unsigned int)(canvas_pos.y / delta_y)};
    if (
        canvas_pos.x < 0.0
        || canvas_pos.y < 0.0
        || canvas_pos.x >= delta_x * gridDim.x * blockDim.x
        || canvas_pos.y >= delta_y * gridDim.y * blockDim.y
    )
    {
        reflection = 0.0;
        canvas_pixel = uint2{0, 0};
    }
    unsigned int canvas_idx = get_array_index_2d(canvas_pixel);
    // TODO: sync all threads
    canvas[canvas_idx] += reflection;
}
