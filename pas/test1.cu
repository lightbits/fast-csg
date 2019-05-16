// nvcc test1.cu -ptx --gpu-architecture=sm_60 -use_fast_math && nvcc test1.cu -cubin --gpu-architecture=sm_60 -use_fast_math && cuobjdump --dump-sass test1.cubin > test1.cuobjdump && nvdisasm -hex test1.cubin > test1.nvdisasm
#include <math_functions.h>
__global__ void test(const float *input, float *output)
{
    const int n = 4;
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    float x = input[i];

    // -0.700000f   -> 0xBF333333
    // -0.69921875f -> 0xBF330000

    output[i] = x;                        i += n;
    output[i] = x + 0.5f;                 i += n;
    output[i] = x + 0.69921875f;          i += n;
    output[i] = x + 0.7f;                 i += n;

    output[i] = abs(x) + 0.5f;            i += n;
    output[i] = abs(x) + 0.69921875f;     i += n;
    output[i] = abs(x) + 0.7f;            i += n;

    output[i] = x*0.5f;                   i += n;
    output[i] = x*0.69921875f;            i += n;
    output[i] = x*0.7f;                   i += n;

    output[i] = max(x, 0.5f);             i += n;
    output[i] = max(x, 0.69921875f);      i += n;
    output[i] = max(x, 0.7f);             i += n;

    output[i] = min(x, 0.5f);             i += n;
    output[i] = min(x, 0.69921875f);      i += n;
    output[i] = min(x, 0.7f);             i += n;

    output[i] = floorf(x);                i += n;
    output[i] = floorf(x*0.5f);           i += n;
    output[i] = __fmul_rd(x,0.5f);        i += n;
}
