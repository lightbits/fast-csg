// nvcc test2.cu -ptx --gpu-architecture=sm_60 -use_fast_math && nvcc test2.cu -cubin --gpu-architecture=sm_60 -use_fast_math && cuobjdump --dump-sass test2.cubin > test2.cuobjdump && nvdisasm -hex test2.cubin > test2.nvdisasm
#include <math_functions.h>
__global__ void test(const float *input, float *output)
{
    const int n = 4;
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    float x = input[i];

    // -0.700000f   -> 0xBF333333
    // -0.69921875f -> 0xBF330000

    output[i] = floorf(x/0.5f);           i += n;
    output[i] = floorf(x/0.69921875f);    i += n;
    output[i] = floorf(x/0.7f);           i += n;

    output[i] = __fdiv_rd(x,0.5f);        i += n;
    output[i] = __fdiv_rd(x,0.69921875f); i += n;
    output[i] = __fdiv_rd(x,0.7f);        i += n;

    output[i] = floorf(__fdividef(x,0.5f));           i += n;
    output[i] = floorf(__fdividef(x,0.69921875f));    i += n;
    output[i] = floorf(__fdividef(x,0.7f));           i += n;

    output[i] = __fdividef(x,0.5f);       i += n;
    output[i] = __fdividef(x,0.69921875f);i += n;
    output[i] = __fdividef(x,0.7f);       i += n;
}
