// Example compilation instructions for Linux, g++:
// (Replace include directory with your installation and version of CUDA)
// $ g++ -std=c++11 backend_ptx.cpp -I/usr/local/cuda-10.1/include -lcuda

#include <iostream>
#include <math.h>
#include <cuda.h>
#include "util/cuda_error.h"
#include "util/init_cuda.h"

#define PTX_FP20_IMMEDIATE
#include "../src/frep.h"
#include "../src/frep_eval.h"
#include "../src/frep_builder.h"
#include "../src/backend_ptx.h"

// This generates a PTX program equivalent to:
//   float tree(float x, float y, float z) {
//       // generated PTX instructions
//   }
//   void main(vec4 *input, float *output) {
//       int tid = threadIdx.x + blockDim.x*blockIdx.x;
//       vec4 p = input[tid];
//       output[tid] = tree(p.x, p.y, p.z);
//   }
// Note: out_length _DOES NOT_ include the null-terminator.
char *generate_ptx_program(frep_t *f, size_t *out_length)
{
    const char *ptx_template = R"str(
    .version 6.0
    .target sm_60
    .address_size 64
    .func (.reg.f32 f%d) tree(.reg.f32 x0, .reg.f32 y0, .reg.f32 z0) {
        .reg.f32 f<%d>;
        %s
        ret.uni;
    }
    .visible.entry main(.param.u64 param0, .param.u64 param1) {
        .reg.f32 x0;
        .reg.f32 y0;
        .reg.f32 z0;
        .reg.f32 w0;
        .reg.b32 r<5>;
        .reg.b64 rd<9>;
        .reg.f32 d;
        ld.param.u64 rd1, [param0];
        ld.param.u64 rd2, [param1];
        cvta.to.global.u64 rd3, rd2;
        cvta.to.global.u64 rd4, rd1;
        mov.u32 r1, %%tid.x;       // threadIdx.x
        mov.u32 r2, %%ctaid.x;     // blockIdx.x
        mov.u32 r3, %%ntid.x;      // blockDim.x
        mad.lo.s32 r4, r3, r2, r1; // blockDim.x*blockIdx.x + threadIdx.x
        mul.wide.s32 rd5, r4, 16;  // sizeof(vec4)*(blockDim.x*blockIdx.x + threadIdx.x)
        add.s64 rd6, rd4, rd5;     // param0 + sizeof(vec4)*(blockDim.x*blockIdx.x + threadIdx.x)
        ld.global.v4.f32 {x0, y0, z0, w0}, [rd6];
        mul.wide.s32 rd7, r4, 4;   // sizeof(float)*(blockDim.x*blockIdx.x + threadIdx.x)
        add.s64 rd8, rd3, rd7;     // param1 + sizeof(float)*(blockDim.x*blockIdx.x + threadIdx.x)
        call.uni (d), tree, (x0,y0,z0);
        st.global.f32 [rd8], d;
        ret;
    }
    )str";

    static char buffer[10*1024*1024];
    char *stream = buffer;
    int result_register;
    char *ptx = generate_ptx(f, &result_register);
    stream += sprintf(stream, ptx_template, result_register, result_register, ptx);
    *out_length = (stream - buffer);
    return buffer;
}

CUmodule load_ptx_program(
    const char *ptx_source, size_t ptx_source_length,
    int jit_optimization_level)
{
    CUmodule module;
    void *cubin; size_t cubin_size;
    CUlinkState link_state;
    enum { num_options = 8 };
    CUjit_option options[num_options];
    void *option_values[num_options];
    float walltime;
    char error_log[8192], info_log[8192];

    assert(jit_optimization_level >= 0 && jit_optimization_level <= 4);

    // see CUDA Driver API manual for these options (look up cuLinkCreate)
    options[0] = CU_JIT_WALL_TIME;                   option_values[0] = (void *) &walltime;
    options[1] = CU_JIT_INFO_LOG_BUFFER;             option_values[1] = (void *) info_log;
    options[2] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;  option_values[2] = (void *) (long)sizeof(info_log);
    options[3] = CU_JIT_ERROR_LOG_BUFFER;            option_values[3] = (void *) error_log;
    options[4] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES; option_values[4] = (void *) (long)sizeof(error_log);
    options[5] = CU_JIT_LOG_VERBOSE;                 option_values[5] = (void *) 1;
    options[6] = CU_JIT_TARGET;                      option_values[6] = (void *) CU_TARGET_COMPUTE_60;
    options[7] = CU_JIT_OPTIMIZATION_LEVEL;          option_values[7] = (void *) (long)jit_optimization_level;
    cudaCheckError(cuLinkCreate(num_options, options, option_values, &link_state));

    int err = cuLinkAddData(link_state, CU_JIT_INPUT_PTX, (void *)ptx_source, ptx_source_length+1, 0, 0, 0, 0);
    if (err != CUDA_SUCCESS)
        fprintf(stderr, "PTX Linker Error:\n%s\n", error_log);
    cudaCheckError(cuLinkComplete(link_state, &cubin, &cubin_size));
    printf("Linking done in %fms. Linker Output:\n%s\n", walltime, info_log);

    cudaCheckError(cuModuleLoadData(&module, cubin)); assert(module);
    cudaCheckError(cuLinkDestroy(link_state));
    return module;
}

void run_ptx_program(
    void *input, size_t sizeof_input,
    void *output, size_t sizeof_output,
    const char *ptx_source, size_t ptx_source_length, const char *entry_name,
    int num_blocks, int threads_per_block, int shared_memory_bytes=1024,
    int jit_optimization_level=1 /*allowed values = 0,1,2,3,4*/)
{
    CUdeviceptr dev_input;
    CUdeviceptr dev_output;
    cudaCheckError(cuMemAlloc(&dev_input, sizeof_input)); assert(dev_input);
    cudaCheckError(cuMemAlloc(&dev_output, sizeof_output)); assert(dev_output);
    cudaCheckError(cuMemcpyHtoD(dev_input, input, sizeof_input));
    CUmodule module = load_ptx_program(ptx_source, ptx_source_length, jit_optimization_level);
    CUfunction kernel = 0;
    cudaCheckError(cuModuleGetFunction(&kernel, module, entry_name));
    uint64_t param0 = (uint64_t)(dev_input);
    uint64_t param1 = (uint64_t)(dev_output);
    void *kernel_params[] = { (void*)&param0, (void*)&param1 };
    cuLaunchKernel(kernel, num_blocks,1,1, threads_per_block,1,1, shared_memory_bytes, NULL, kernel_params, NULL);
    cudaCheckError(cuCtxSynchronize());
    cudaCheckError(cuMemcpyDtoH(output, dev_output, sizeof_output));
    cudaCheckError(cuMemFree(dev_output));
    cudaCheckError(cuMemFree(dev_input));
    cudaCheckError(cuModuleUnload(module));
}

void run_test(int test_number, frep_t *f)
{
    printf("///////////////////////////////////////////////////\n");
    printf("            running test number %d\n", test_number);

    const int num_points_x = 4;
    const int num_points_y = 4;
    const int num_points_z = 4;
    const int num_threads = 32;
    const int num_points = num_points_x*num_points_y*num_points_z;
    const int num_blocks = num_points/num_threads;
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
        run_ptx_program(
            input, sizeof_input,
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
    init_cuda();

    frep_t *f = fBoxCheap(1.0f, 0.5f, 0.25f);
    run_test(1, f);

    return 0;
}
