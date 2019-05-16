// nvcc test3.cu -ptx --gpu-architecture=sm_60 -use_fast_math && nvcc test3.cu -cubin --gpu-architecture=sm_60 -use_fast_math && cuobjdump --dump-sass test3.cubin > test3.cuobjdump && nvdisasm -hex test3.cubin > test3.nvdisasm

// In this test I wanted to find out how many bits are allowed with immediate values.
// Some instructions have 32-bit immediate versions (FADD32I and FMUL32I), but others
// do not (FMNMX). However, 32-bit fp instructions can sometimes store a 32-bit float
// if part of the mantissa is zero. For example, 1 sign bit, 8 exponent bits and 7
// mantissa bits can go into all FADD, FMUL, FMNMX instructions. This is not a 16-bit
// half-precision float, so I'm wondering how many bits are actually allowed to fit.

#include <math_functions.h>
__global__ void test(const float *input, float *output)
{
    const int n = 4;
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    float x = input[i];
    float y = input[i+1024*1024];

    // -0.700000f   -> 0xBF333333
    // -0.69921875f -> 0xBF330000
    // 1.99951171875f -> 0x3FFFF000

    // 0.562255859375f -> 0x3F0FF000

    output[i] = x;                        i += n;
    output[i] = x + 0.5f;                 i += n;
    output[i] = x + 0.69921875f;          i += n;
    output[i] = x + 1.99951171875f;       i += n;
    output[i] = x - 0.5f;                 i += n;
    output[i] = x - 0.69921875f;          i += n;
    output[i] = x - 1.99951171875f;       i += n;

    output[i] = x*(+0.5f);                 i += n;
    output[i] = x*(+0.69921875f);          i += n;
    output[i] = x*(+1.99951171875f);       i += n;
    output[i] = x*(-0.5f);                 i += n;
    output[i] = x*(-0.69921875f);          i += n;
    output[i] = x*(-1.99951171875f);       i += n;

    output[i] = x*(+0.25f) + y;                i += n;
    output[i] = x*(+0.562255859375f) + y;      i += n;
    output[i] = x*(+0.9f) + y;                 i += n;
    output[i] = x*(-0.25f) + y;                i += n;
    output[i] = x*(-0.9f) + y;                 i += n;

    output[i] = max(x, +0.5f);                 i += n;
    output[i] = max(x, +0.69921875f);          i += n;
    output[i] = max(x, +1.99951171875f);       i += n;
    output[i] = max(x, -0.5f);                 i += n;
    output[i] = max(x, -0.69921875f);          i += n;
    output[i] = max(x, -1.99951171875f);       i += n;

    output[i] = min(x, +0.5f);                 i += n;
    output[i] = min(x, +0.69921875f);          i += n;
    output[i] = min(x, +1.99951171875f);       i += n;
    output[i] = min(x, -0.5f);                 i += n;
    output[i] = min(x, -0.69921875f);          i += n;
    output[i] = min(x, -1.99951171875f);       i += n;
}
