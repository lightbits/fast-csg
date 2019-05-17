#define PTX_FP20_IMMEDIATE
#include <iostream>
#include <math.h>
#include "../src/frep.h"
#include "../src/frep_builder.h"
#include "../src/backend_ptx.h"
#include "util/test_models.h"
#include "util/ptx_template.h"
#include "util/ptxjit.h"

/*
This generates a PTX program equivalent to

  float f(float x, float y, float z);
  void main(vec4 *input, float *output) {
      int tid = threadIdx.x + blockDim.x*blockIdx.x;
      vec4 p = input[tid];
      output[tid] = f(p.x, p.y, p.z);
  }

Note: See util/ptx_template.h for details.
Note: out_length _DOES NOT_ include the null-terminator.
*/
char *generate_ptx_program(frep_t *f, size_t *out_length)
{
    static char buffer[10*1024*1024];
    char *stream = buffer;
    int result_register;
    char *ptx = generate_ptx(f, &result_register);
    stream += sprintf(stream, ptx_template, result_register, result_register, ptx);
    *out_length = (stream - buffer);
    return buffer;
}

void run_test(int test_number, frep_t *f)
{
    printf("///////////////////////////////////////////////////\n");
    printf("                 test number %d\n", test_number);

    const int num_points_x = 4;
    const int num_points_y = 4;
    const int num_points_z = 4;
    const int num_threads = 32;
    const int num_blocks = num_points/num_threads;
    const int num_points = num_points_x*num_points_y*num_points_z;
    const int sizeof_input = num_points*4*sizeof(float);
    const int sizeof_output = num_points*1*sizeof(float);

    float *output = (float*)malloc(sizeof_output); assert(output);
    float *cpu_output = (float*)malloc(sizeof_output); assert(cpu_output);
    float *input = (float*)malloc(num_points*4*sizeof(float));

    // generate input array data (points sampled in regular grid)
    {
        float *p = input;
        for (int zi = 0; zi < num_points_z; zi++)
        for (int yi = 0; yi < num_points_y; yi++)
        for (int xi = 0; xi < num_points_x; xi++)
        {
            p[0] = (-1.0f + 2.0f*xi/num_points_x);
            p[1] = (-1.0f + 2.0f*yi/num_points_y);
            p[2] = (-1.0f + 2.0f*zi/num_points_z);
            p[3] = 0.0f;
            p += 4;
        }
    }

    // compute expected output using CPU-based evaluator
    {
        for (int i = 0; i < num_points; i++)
        {
            float x = input[4*i + 0];
            float y = input[4*i + 1];
            float z = input[4*i + 2];
            cpu_output[i] = frep_eval(f, x, y, z);
        }
    }

    // compute output using GPU
    {
        size_t ptx_length;
        char *ptx_source = generate_ptx_program(f, &ptx_length);
        ptxjit(input, sizeof_input,
               output, sizeof_output,
               ptx_source, ptx_length,
               "main",
               num_blocks, num_threads);
    }

    // verify that GPU output matches CPU output
    for (int i = 0; i < num_points; i++)
    {
        float d_cpu = cpu_output[i];
        float d_ptx = output[i];
        if (fabsf(d_cpu - d_ptx) > 0.01f)
        {
            float x = input[4*i + 0];
            float y = input[4*i + 1];
            float z = input[4*i + 2];
            printf("\nEvaluation mismatch!\n");
            printf("cpu: f(%.2f,%.2f,%.2f) = %f\n", x, y, z, d_cpu);
            printf("ptx: f(%.2f,%.2f,%.2f) = %f\n", x, y, z, d_ptx);
            exit(1);
        }
    }

    free(output);
    free(cpu_output);
    free(input);
}

int main(int argc, char **argv)
{
    frep_t *f = fBoxCheap(1.0f, 0.5f, 0.25f);
    run_test(f);

    return 0;
}
