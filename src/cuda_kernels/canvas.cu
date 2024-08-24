#include "cuda_kernels/common.cu"

extern "C" __global__
void canvas(
    double* heightmap,
    double* canvas,
    double delta_x,
    double delta_y,
    double distance // Mirror distance from the canvas
)
{
    uint2 mirror_pixel = get_voxel_coords_2d();
    // The position takes the (0, 0) value in the upper left corner of the mirror viewed from the canvas.
    // We use a right-handed coordinate system
    double3 mirror_pos = double3{(double)mirror_pixel.x * delta_x, (double)mirror_pixel.y * delta_y, distance};
    unsigned int idx = get_array_index_2d();
    double3 canvas_normal = double3{0.0, 0.0, 1.0};
    double3 canvas_point = double3{0.0, 0.0, 0.0};
    double height = heightmap[idx];
    mirror_pos.z -= height; // Correct for height differences

    // Mirror normal vector (central difference approximation):
    uint2 dimensions = uint2{ gridDim.x * blockDim.x, gridDim.y * blockDim.y };
    uint2 mirror_pixel_p0 = uint2{mirror_pixel.x + 1, mirror_pixel.y};
    if (mirror_pixel.x == dimensions.x - 1)
        mirror_pixel_p0.x = dimensions.x - 1;
    unsigned int idx_p0 = get_array_index_2d(mirror_pixel_p0);

    uint2 mirror_pixel_n0 = uint2{mirror_pixel.x - 1, mirror_pixel.y};
    if (mirror_pixel.x == 0)
        mirror_pixel_n0.x = 0;
    unsigned int idx_n0 = get_array_index_2d(mirror_pixel_n0);

    uint2 mirror_pixel_0p = uint2{mirror_pixel.x, mirror_pixel.y + 1};
    if (mirror_pixel.y == dimensions.y - 1)
        mirror_pixel_0p.y = dimensions.y - 1;
    unsigned int idx_0p = get_array_index_2d(mirror_pixel_0p);

    uint2 mirror_pixel_0n = uint2{mirror_pixel.x, mirror_pixel.y - 1};
    if (mirror_pixel.y == 0)
        mirror_pixel_0n.y = 0;
    unsigned int idx_0n = get_array_index_2d(mirror_pixel_0n);

    // Calculate central difference:
    double2 grad = double2{
        (heightmap[idx_p0] - heightmap[idx_n0]) / (2.0 * delta_x),
        (heightmap[idx_0p] - heightmap[idx_0n]) / (2.0 * delta_y)
    };

    // calculate mirror normal vector:
    double3 tangent = double3{delta_x, 0.0, grad.x};
    double3 bitangent = double3{0.0, delta_y, grad.y};
    // The normal of the mirror should point towards the canvas:
    double3 mirror_normal = normalize(cross(bitangent, tangent));

    // Calculate reflected light and canvas position:
    // Use mirror_pos and normal
    double reflection = 1.0; // Test
    double t = 0.0;
    bool is_intersect = intersectPlane(canvas_normal, canvas_point, mirror_pos, reflect(mirror_normal, canvas_normal), t);
    double3 canvas_pos = double3{0.0, 0.0, 0.0};
    if (is_intersect) {
        canvas_pos = mirror_pos + t * mirror_normal;
    }
    else {
        reflection = 0.0;    // No reflection when no intersection between the canvas and the ray
    }

    uint2 canvas_pixel = uint2{(unsigned int)(canvas_pos.x / delta_x), (unsigned int)(canvas_pos.y / delta_y)};
    // Check whether the coordinates are on the rendered canvas
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
    atomicAdd(&canvas[canvas_idx], reflection);
}
