#ifndef CUDA_COMMON_INCLUDE
#define CUDA_COMMON_INCLUDE

#include <cupy/complex.cuh>

constexpr float M_PI_f = 3.14159265358979323846264338327950288419716939937510f;
constexpr double M_PI_d = 3.14159265358979323846264338327950288419716939937510;
constexpr float sqrt_2 = 1.414213562373095f;
constexpr float sqrt_3 = 1.732050807568877f;

// Vector concepts:
template<typename T4>
concept vec4 = requires (const T4& v) {
    v.x; v.y; v.z; v.w;
};

template<typename T3>
concept vec3 = requires (const T3& v) {
    v.x; v.y; v.z;
} && !vec4<T3>;

template<typename T2>
concept vec2 = requires (const T2& v) {
    v.x; v.y;
} && !vec3<T2> && ! vec4<T2>;

template<vec3 T3>
__device__ T3 abs(const T3& v)
{
    return {fabsf(v.x), fabsf(v.y), fabsf(v.z)};
}

template<typename T>
__device__ complex<T> mul(const complex<T>& a, const complex<T>& b)
{
    return complex<T>(a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real() );
}

template<typename T, vec3 T3>
__device__ constexpr T3 scalarVectorMul(const T s, const T3& v)
{
    return {s * v.x, s * v.y, s * v.z};
}

template<typename T, vec3 T3>
__device__ constexpr T3 operator*(T s, const T3& v)
{
    return {s * v.x, s * v.y, s * v.z};
}

template<typename T, vec3 T3>
__device__ constexpr T3 operator*(const T3& v, T s)
{
    return {s * v.x, s * v.y, s * v.z};
}

template<typename T, vec2 T2>
__device__ constexpr T2 operator*(const T2& v, T s)
{
    return {s * v.x, s * v.y};
}

template<vec3 T3>
__device__ constexpr T3 operator-(const T3& v)
{
    return {-v.x, -v.y, -v.z};
}

template<vec2 T2>
__device__ constexpr T2 operator-(const T2& v)
{
    return {-v.x, -v.y};
}

constexpr __device__ float dot(const float3& a, const float3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

constexpr __device__ float dot(const float4& a, const float4& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

template<vec3 T3>
constexpr __device__ T3 cross(const T3& a, const T3& b)
{
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

constexpr __device__ double dot(const double3& a, const double3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

constexpr __device__ double dot(const double4& a, const double4& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__device__ const complex<float> exp_i(float angle)
{
    return complex<float>(cosf(angle), sinf(angle));
}

__device__ const complex<float> cexp_i(const complex<float>& cangle)
{
    return complex<float>(cosf(cangle.real()), sinf(cangle.real())) * expf(-cangle.imag());
}

__device__ const complex<double> exp_i(double angle)
{
    return complex<double>(cos(angle), sin(angle));
}

__device__ const complex<double> cexp_i(const complex<double>& cangle)
{
    return complex<double>(cos(cangle.real()), sin(cangle.real())) * exp(-cangle.imag());
}

__device__ constexpr float3 add(const float3& a, const float3& b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

template<vec4 T4>
__device__ constexpr T4 operator+(const T4& a, const T4& b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}

template<vec4 T4, typename T>
__device__ constexpr T4 operator+(const T4& a, T b)
{
    return {a.x + b.x, a.y + b, a.z + b, a.w + b};
}

template<typename T, vec4 T4>
__device__ constexpr T4 operator+(T a, const T4& b)
{
    return {a + b.x, a + b.y, a + b.z, a + b.w};
}

template<vec3 T3>
__device__ constexpr T3 operator+(const T3& a, const T3& b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

template<vec3 T3, typename T>
__device__ constexpr T3 operator+(const T3& a, T b)
{
    return {a.x + b, a.y + b, a.z + b};
}

template<typename T, vec3 T3>
__device__ constexpr T3 operator+(T a, const T3& b)
{
    return {a + b.x, a + b.y, a + b.z};
}

template<vec2 T2>
__device__ constexpr T2 operator+(const T2& a, const T2& b)
{
    return {a.x + b.x, a.y + b.y};
}

template<vec2 T2, typename T>
__device__ constexpr T2 operator+(const T2& a, T b)
{
    return {a.x + b, a.y + b};
}

template<typename T, vec2 T2>
__device__ constexpr T2 operator+(T a, const T2& b)
{
    return {a + b.x, a + b.y};
}

/*
template<typename T>
__device__ constexpr complex<T> operator+(T r, const complex<T>& c)
{
    return complex<T>(r) + c;
}

template<typename T>
__device__ constexpr complex<T> operator+(const complex<T>& c, T r)
{
    return c + complex<T>(r);
}
*/

__device__ float3& operator+=(float3& a, const float3& b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
    return a;
}

__device__ constexpr double3 add(const double3& a, const double3& b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__device__ double3& operator+=(double3& a, const double3& b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
    return a;
}

template<vec3 T3>
__device__ constexpr T3 diff(const T3& a, const T3& b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

template<vec4 T4>
__device__ constexpr T4 operator-(const T4& a, const T4& b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
}

template<vec4 T4, typename T>
__device__ constexpr T4 operator-(const T4& a, T b)
{
    return {a.x - b, a.y - b, a.z - b, a.w - b};
}

template<typename T, vec4 T4>
__device__ constexpr T4 operator-(T a, const T4& b)
{
    return {a - b.x, a - b.y, a - b.z, a - b.w};
}

template<vec3 T3>
__device__ constexpr T3 operator-(const T3& a, const T3& b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

template<vec3 T3, typename T>
__device__ constexpr T3 operator-(const T3& a, T b)
{
    return {a.x - b, a.y - b, a.z - b};
}

template<typename T, vec3 T3>
__device__ constexpr T3 operator-(T a, const T3& b)
{
    return {a - b.x, a - b.y, a - b.z};
}

template<vec2 T2>
__device__ constexpr T2 operator-(const T2& a, const T2& b)
{
    return {a.x - b.x, a.y - b.y};
}

template<vec2 T2, typename T>
__device__ constexpr T2 operator-(const T2& a, T b)
{
    return {a.x - b, a.y - b};
}

template<typename T, vec2 T2>
__device__ constexpr T2 operator-(T a, const T2& b)
{
    return {a - b.x, a - b.y};
}

__device__ float3 mul(const float3& a, const float3& b)
{
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

__device__ float2 mul(const float2& a, const float2& b)
{
    return {a.x * b.x, a.y * b.y};
}

__device__ float3 operator*(const float3& a, const float3& b)
{
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

__device__ float2 operator*(const float2& a, const float2& b)
{
    return {a.x * b.x, a.y * b.y};
}

__device__ double3 mul(const double3& a, const double3& b)
{
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

__device__ double2 mul(const double2& a, const double2& b)
{
    return {a.x * b.x, a.y * b.y};
}

__device__ double3 operator*(const double3& a, const double3& b)
{
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

__device__ double2 operator*(const double2& a, const double2& b)
{
    return {a.x * b.x, a.y * b.y};
}

template<vec3 T3>
__device__ constexpr T3 div(const T3& a, const T3& b)
{
    return {a.x / b.x, a.y / b.y, a.z / b.z};
}

template<vec4 T4, typename T>
__device__ constexpr T4 operator/(const T4& a, T b)
{
    return {a.x / b, a.y / b, a.z / b, a.w / b};
}

template<vec4 T4>
__device__ constexpr T4 operator/(const T4& a, const T4& b)
{
    return {a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w};
}

template<vec3 T3>
__device__ constexpr T3 operator/(const T3& a, const T3& b)
{
    return {a.x / b.x, a.y / b.y, a.z / b.z};
}

template<vec2 T2>
__device__ constexpr T2 operator/(const T2& a, const T2& b)
{
    return {a.x / b.x, a.y / b.y};
}

template<vec3 T3, typename T>
__device__ constexpr T3 operator/(const T3& v, T s)
{
    return {v.x / s, v.y / s, v.z / s};
}

__device__ float length(const float3& a)
{
    return sqrtf(dot(a, a));
}

__device__ float3 normalize(const float3& a)
{
    return a / sqrtf(dot(a, a));
}

__device__ float4 normalize(const float4& a)
{
    return a / sqrtf(dot(a, a));
}

__device__ double length(const double3& a)
{
    return sqrt(dot(a, a));
}

__device__ double3 normalize(const double3& a)
{
    return a / sqrt(dot(a, a));
}

__device__ double4 normalize(const double4& a)
{
    return a / sqrt(dot(a, a));
}

template<vec3 T3, typename T>
__device__ constexpr T3 transform_corner_origin_to_center_origin_system(const T3& pos)
{
    return pos - T3{(T)(gridDim.x * blockDim.x), (T)(gridDim.y * blockDim.y), (T)(gridDim.z * blockDim.z)} / (T)2;
}

__device__ constexpr double3 transform_corner_origin_to_center_origin_system(const double3& pos)
{
    return pos - 0.5 * double3{(double)(gridDim.x * blockDim.x), (double)(gridDim.y * blockDim.y), (double)(gridDim.z * blockDim.z)};
}

__device__ constexpr uint3 get_voxel_count_3d()
{
    return {
        gridDim.x * blockDim.x,
        gridDim.y * blockDim.y,
        gridDim.z * blockDim.z
    };
}

__device__ constexpr uint2 get_voxel_count_2d()
{
    return {
        gridDim.x * blockDim.x,
        gridDim.y * blockDim.y
    };
}

__device__ uint3 get_voxel_coords()
{
    return {
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y,
        blockIdx.z * blockDim.z + threadIdx.z
    };
}

__device__ uint2 get_voxel_coords_2d()
{
    return {
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y
    };
}

__device__ uint3 get_voxel_coords_inverted()
{
    unsigned int x = gridDim.x * blockDim.x - (blockIdx.x * blockDim.x + threadIdx.x) - 1;
    unsigned int y = gridDim.y * blockDim.y - (blockIdx.y * blockDim.y + threadIdx.y) - 1;
    unsigned int z = gridDim.z * blockDim.z - (blockIdx.z * blockDim.z + threadIdx.z) - 1;
    return {x, y, z};
}

__device__ unsigned int get_array_index()
{
    uint3 voxel = get_voxel_coords();
    return voxel.x * gridDim.y * blockDim.y * gridDim.z * blockDim.z
            + voxel.y * gridDim.z * blockDim.z
            + voxel.z;
}

__device__ unsigned int get_array_index(const uint3& voxel)
{
    return voxel.x * gridDim.y * blockDim.y * gridDim.z * blockDim.z
            + voxel.y * gridDim.z * blockDim.z
            + voxel.z;
}

__device__ unsigned int get_array_index(const uint3& voxel, const uint3& N)
{
    return voxel.x * N.y * N.z
            + voxel.y * N.z
            + voxel.z;
}

__device__ unsigned int get_array_index_2d()
{
    uint2 pixel = get_voxel_coords_2d();
    return pixel.x * gridDim.y * blockDim.y
            + pixel.y;
}

__device__ unsigned int get_array_index_2d(const uint2& pixel)
{
    return pixel.x * gridDim.y * blockDim.y
            + pixel.y;
}

__device__ unsigned int get_array_index_inverted()
{
    uint3 voxel = get_voxel_coords_inverted();
    return voxel.x * gridDim.y * blockDim.y * gridDim.z * blockDim.z
            + voxel.y * gridDim.z * blockDim.z
            + voxel.z;
}


__device__ float3 operator*(const float (&m)[3][3], const float3& v)
{
    return {
        m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
        m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
        m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z
    };
}

__device__ double3 operator*(const double (&m)[3][3], const double3& v)
{
    return {
        m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
        m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
        m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z
    };
}

__device__ float4 operator*(const float (&m)[4][4], const float4& v)
{
    return {
        m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z + m[0][3] * v.w,
        m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z + m[1][3] * v.w,
        m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z + m[2][3] * v.w,
        m[3][0] * v.x + m[3][1] * v.y + m[3][2] * v.z + m[3][3] * v.w,
    };
}

__device__ float4 matVecMul(const float* m, const float4& v)
{
    constexpr size_t size = sizeof(float);
    return {
        m[0 * size + 0] * v.x + m[0 * size + 1] * v.y + m[0 * size + 2] * v.z + m[0 * size + 3] * v.w,
        m[1 * size + 0] * v.x + m[1 * size + 1] * v.y + m[1 * size + 2] * v.z + m[1 * size + 3] * v.w,
        m[2 * size + 0] * v.x + m[2 * size + 1] * v.y + m[2 * size + 2] * v.z + m[2 * size + 3] * v.w,
        m[3 * size + 0] * v.x + m[3 * size + 1] * v.y + m[3 * size + 2] * v.z + m[3 * size + 3] * v.w,
    };
}

__device__ double4 operator*(const double (&m)[4][4], const double4& v)
{
    return {
        m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z + m[0][3] * v.w,
        m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z + m[1][3] * v.w,
        m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z + m[2][3] * v.w,
        m[3][0] * v.x + m[3][1] * v.y + m[3][2] * v.z + m[3][3] * v.w,
    };
}

template<typename T>
__device__ T map(T x, T fromMin, T fromMax, T toMin, T toMax)
{
    return toMin + (toMax - toMin) * (x - fromMin) / (fromMax - fromMin);
}

__device__ float3 rotate_vector(const float3& v, const float3& axis, float rad)
{
    float q0 = cosf(rad / 2.0f);
    float q1 = sinf(rad / 2.0f) * axis.x;
    float q2 = sinf(rad / 2.0f) * axis.y;
    float q3 = sinf(rad / 2.0f) * axis.z;
    float Q[3][3] = { { 0.0f } }; // 3x3 rotation matrix

    Q[0][0] = q0*q0 + q1*q1 - q2*q2 - q3*q3;
    Q[0][1] = 2.0f * (q1*q2 - q0*q3);
    Q[0][2] = 2.0f * (q1*q3 + q0*q2);

    Q[1][0] = 2.0f * (q2*q1 + q0*q3);
    Q[1][1] = q0*q0 - q1*q1 + q2*q2 - q3*q3;
    Q[1][2] = 2.0f * (q2*q3 - q0*q1);

    Q[2][0] = 2.0f * (q3*q1 - q0*q2);
    Q[2][1] = 2.0f * (q3*q2 + q0*q1);
    Q[2][2] = q0*q0 - q1*q1 - q2*q2 + q3*q3;

    return Q * v;
}

__device__ double3 rotate_vector(const double3& v, const double3& axis, double rad)
{
    double q0 = cosf(rad / 2.0);
    double q1 = sinf(rad / 2.0) * axis.x;
    double q2 = sinf(rad / 2.0) * axis.y;
    double q3 = sinf(rad / 2.0) * axis.z;
    double Q[3][3] = { { 0.0 } }; // 3x3 rotation matrix

    Q[0][0] = q0*q0 + q1*q1 - q2*q2 - q3*q3;
    Q[0][1] = 2.0 * (q1*q2 - q0*q3);
    Q[0][2] = 2.0 * (q1*q3 + q0*q2);

    Q[1][0] = 2.0 * (q2*q1 + q0*q3);
    Q[1][1] = q0*q0 - q1*q1 + q2*q2 - q3*q3;
    Q[1][2] = 2.0 * (q2*q3 - q0*q1);

    Q[2][0] = 2.0 * (q3*q1 - q0*q2);
    Q[2][1] = 2.0 * (q3*q2 + q0*q1);
    Q[2][2] = q0*q0 - q1*q1 - q2*q2 + q3*q3;

    return Q * v;
}

__device__ float mix(const float3& xyz, float u, float v)
{
    return xyz.z * v + (1.0f - v) * (xyz.y * u + xyz.x * (1.0f - u));
}

__device__ double mix(const double3& xyz, double u, double v)
{
    return xyz.z * v + (1.0 - v) * (xyz.y * u + xyz.x * (1.0 - u));
}

__device__ unsigned int get_block_local_idx_3d()
{
    return threadIdx.x * blockDim.y * blockDim.z
            + threadIdx.y * blockDim.z
            + threadIdx.z;
}

__device__ unsigned int get_block_local_idx_2d()
{
    return threadIdx.x * blockDim.y
            + threadIdx.y;
}

template<typename T>
__device__ T get_simpson_coefficient_3d(const uint3& voxel, const uint3& sample_count);

template<>
__device__ float get_simpson_coefficient_3d<float>(const uint3& voxel, const uint3& sample_count)
{
    uint3 n = get_voxel_count_3d();    // In the integrated volume
    float sX = 1.0f;
    if (voxel.x > 0 && voxel.x < sample_count.x - 1) {
        if (voxel.x % 2 == 0)
            sX = 2.0f;
        else
            sX = 4.0f;
    }
    float sY = 1.0f;
    if (voxel.y > 0 && voxel.y < sample_count.y - 1) {
        if (voxel.y % 2 == 0)
            sY = 2.0f;
        else
            sY = 4.0f;
    }
    float sZ = 1.0f;
    if (voxel.z > 0 && voxel.z < sample_count.z - 1) {
        if (voxel.z % 2 == 0)
            sZ = 2.0f;
        else
            sZ = 4.0f;
    }
    return sX * sY * sZ / 27.0f;    // ... / 3^3
}

template<>
__device__ double get_simpson_coefficient_3d<double>(const uint3& voxel, const uint3& sample_count)
{
    uint3 n = get_voxel_count_3d();    // In the integrated volume
    double sX = 1.0;
    if (voxel.x > 0 && voxel.x < sample_count.x - 1) {
        if (voxel.x % 2 == 0)
            sX = 2.0;
        else
            sX = 4.0;
    }
    double sY = 1.0;
    if (voxel.y > 0 && voxel.y < sample_count.y - 1) {
        if (voxel.y % 2 == 0)
            sY = 2.0;
        else
            sY = 4.0;
    }
    double sZ = 1.0;
    if (voxel.z > 0 && voxel.z < sample_count.z - 1) {
        if (voxel.z % 2 == 0)
            sZ = 2.0;
        else
            sZ = 4.0;
    }
    return sX * sY * sZ / 27.0;    // ... / 3^3
}

template<typename T>
__device__ T get_simpson_coefficient_2d(const uint2& pixel, const uint2& sample_count);

template<>
__device__ float get_simpson_coefficient_2d<float>(const uint2& pixel, const uint2& sample_count)
{
    float sX = 1.0f;
    if (pixel.x > 0 && pixel.x < sample_count.x - 1) {
        if (pixel.x % 2 == 0)
            sX = 2.0f;
        else
            sX = 4.0f;
    }
    float sY = 1.0f;
    if (pixel.y > 0 && pixel.y < sample_count.y - 1) {
        if (pixel.y % 2 == 0)
            sY = 2.0f;
        else
            sY = 4.0f;
    }
    return sX * sY / 9.0f;    // ... / 3^2
}

template<>
__device__ double get_simpson_coefficient_2d<double>(const uint2& pixel, const uint2& sample_count)
{
    double sX = 1.0;
    if (pixel.x > 0 && pixel.x < sample_count.x - 1) {
        if (pixel.x % 2 == 0)
            sX = 2.0;
        else
            sX = 4.0;
    }
    double sY = 1.0;
    if (pixel.y > 0 && pixel.y < sample_count.y - 1) {
        if (pixel.y % 2 == 0)
            sY = 2.0;
        else
            sY = 4.0;
    }
    return sX * sY / 9.0;    // ... / 3^2
}

template<typename T, typename T3>
__device__ complex<T> gaussian_wave_packet(const T3& sigma, const T3& r, const T3& r_0, const T3& k_0);


template<>
__device__ complex<float> gaussian_wave_packet<float, float3>(const float3& sigma, const float3& r, const float3& r_0, const float3& k_0)
{
    float3 a = 2.0f * sigma;
    float g_x = powf(2.0f / M_PI_f / a.x / a.x, 1.0f / 4.0f)
        * expf(
            -(r.x - r_0.x) * (r.x - r_0.x) / a.x / a.x
        );
    float g_y = powf(2.0f / M_PI_f / a.y / a.y, 1.0f / 4.0f)
        * expf(
            -(r.y - r_0.y) * (r.y - r_0.y) / a.y / a.y
        );
    float g_z = powf(2.0f / M_PI_f / a.z / a.z, 1.0f / 4.0f)
        * expf(
            -(r.z - r_0.z) * (r.z - r_0.z) / a.z / a.z
        );
    return g_x * g_y * g_z * exp_i(dot(k_0, r));
}

template<>
__device__ complex<double> gaussian_wave_packet<double, double3>(const double3& sigma, const double3& r, const double3& r_0, const double3& k_0)
{
    double3 a = 2.0 * sigma;
    double g_x = pow(2.0 / M_PI_d / a.x / a.x, 1.0 / 4.0)
        * exp(
            -(r.x - r_0.x) * (r.x - r_0.x) / a.x / a.x
        );
    double g_y = pow(2.0 / M_PI_d / a.y / a.y, 1.0 / 4.0)
        * exp(
            -(r.y - r_0.y) * (r.y - r_0.y) / a.y / a.y
        );
    double g_z = pow(2.0 / M_PI_d / a.z / a.z, 1.0 / 4.0)
        * exp(
            -(r.z - r_0.z) * (r.z - r_0.z) / a.z / a.z
        );
    return g_x * g_y * g_z * exp_i(dot(k_0, r));
}

template<typename T, typename T2, typename T3>
__device__ complex<T> truncated_plain_wave(const T2& rectangleBottomCorner, const T2& rectangleTopCorner, const T3& sigma, const T3& r, const T3& r_0, const T3& k_0, const T3& normal);

template<>
__device__ complex<float> truncated_plain_wave<float, float2, float3>(const float2& rectangleBottomCorner, const float2& rectangleTopCorner, const float3& sigma, const float3& r, const float3& r_0, const float3& k_0, const float3& normal)
{
    float3 a = 2.0f * sigma;
    float3 prefUp = {0.0f, 1.0f, 0.0f};
    if (fabsf(normal.y) > 0.99f) {
        prefUp.y = 0.0f;
        prefUp.z = 1.0f;
    }
    const float3 right = normalize(cross(normal, prefUp));
    const float3 up = normalize(cross(right, normal));

    float r0x = dot(right, r_0);
    float r0y = dot(up, r_0);
    float ax = dot(right, a);
    float ay = dot(up, a);
    float kx = dot(right, k_0);
    float ky = dot(up, k_0);
    float rx = dot(right, r);
    float ry = dot(up, r);

    complex<float> conv = (erff( ( rx - rectangleBottomCorner.x ) / ax ) - erff( ( rx - rectangleTopCorner.x ) / ax ))
        *(erff( ( ry - rectangleBottomCorner.y ) / ay ) - erff( ( ry - rectangleTopCorner.y ) / ay ));
    float rz = dot(normal, r);
    return conv * cexp_i(complex<float>{dot(normal, k_0) * rz, powf(rz - dot(normal, r_0), 2) / powf(dot(normal, a), 2)});
}

template<>
__device__ complex<double> truncated_plain_wave<double, double2, double3>(
    const double2& rectangleBottomCorner,
    const double2& rectangleTopCorner,
    const double3& sigma,
    const double3& r,
    const double3& r_0,
    const double3& k_0,
    const double3& normal
)
{
    double3 a = 2.0 * sigma;
    double3 prefUp = {0.0, 1.0, 0.0};
    if (abs(normal.y) > 0.99) {
        prefUp.y = 0.0;
        prefUp.z = 1.0;
    }
    const double3 right = normalize(cross(normal, prefUp));
    const double3 up = normalize(cross(right, normal));

    double r0x = dot(right, r_0);
    double r0y = dot(up, r_0);
    double ax = dot(right, a);
    double ay = dot(up, a);
    double kx = dot(right, k_0);
    double ky = dot(up, k_0);
    double rx = dot(right, r);
    double ry = dot(up, r);

    complex<double> conv = (erf( ( rx - rectangleBottomCorner.x ) / ax ) - erf( ( rx - rectangleTopCorner.x ) / ax ))
        *(erf( ( ry - rectangleBottomCorner.y ) / ay ) - erf( ( ry - rectangleTopCorner.y ) / ay ));
    double rz = dot(normal, r);
    return conv * cexp_i(complex<double>{dot(normal, k_0) * rz, pow(rz - dot(normal, r_0), 2) / pow(dot(normal, a), 2)});
}


/*
Parallel Reduction with Sequential Addressing
Based on the "Reduction #3: Sequential Addressing" code published in Mark Harris's Optimizing Parallel Reduction in CUDA presentation slides
URL: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
*/
template<typename T>
__device__ void parallel_reduction_sequential(unsigned int threadId, T sdata[])
{
    __syncthreads();
    unsigned int blockSize = blockDim.x * blockDim.y * blockDim.z;
    for (unsigned int s=1; s < blockSize; s *= 2) { // Reduction #3: Sequential Addressing from Optimizing Parallel Reduction in CUDA by Mark Harris NVIDIA
        int index = 2 * s * threadId;
        if (index + s < blockSize) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }
}

/*
Parallel Reduction with Sequential Addressing
Based on the "Reduction #3: Sequential Addressing" code published in Mark Harris's Optimizing Parallel Reduction in CUDA presentation slides
URL: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
*/
template<typename T>
__device__ void parallel_reduction_sequential_2d(unsigned int threadId, T sdata[])
{
    __syncthreads();
    unsigned int blockSize = blockDim.x * blockDim.y;
    for (unsigned int s=1; s < blockSize; s *= 2) { // Reduction #3: Sequential Addressing from Optimizing Parallel Reduction in CUDA by Mark Harris NVIDIA
        int index = 2 * s * threadId;
        if (index + s < blockSize) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }
}

template<vec3 T3>
__device__ T3 position_operator(const uint3& voxel, const T3& delta_r, const T3& bottom_corner)
{
    return T3{
        bottom_corner.x + delta_r.x * voxel.x,
        bottom_corner.y + delta_r.y * voxel.y,
        bottom_corner.z + delta_r.z * voxel.z
    };
}

/*
    inv_position_operator calculates the voxel from a given position
*/
template<vec3 T3>
__device__ uint3 inv_position_operator(const T3& r, const T3& delta_r, const T3& bottom_corner)
{
    return T3{
        (unsigned int)((r.x - bottom_corner.x) / delta_r.x),
        (unsigned int)((r.y - bottom_corner.y) / delta_r.y),
        (unsigned int)((r.z - bottom_corner.z) / delta_r.z),
    };
}

#endif  // CUDA_COMMON
