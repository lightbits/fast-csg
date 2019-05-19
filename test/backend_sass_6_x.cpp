#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include "util/cuda_error.h"
#include "util/init_cuda.h"
#include "../src/frep.h"
#include "../src/frep_eval.h"
#include "../src/frep_builder.h"
#include "../src/backend_sass.h"

CUmodule link_sass(CUmodule *module,
                   void *cubin1, size_t sizeof_cubin1,
                   void *cubin2, size_t sizeof_cubin2);

int main(int argc, char **argv)
{
    setenv("CUDA_CACHE_DISABLE", "1", 1);
    init_cuda();

    system("/usr/local/cuda-10.1/bin/nvcc "
           "--gpu-architecture=sm_60 "
           "--cubin "
           "--relocatable-device-code=true "
           "main.cu "
           "--output-file main.cubin");

    size_t sizeof_cubin_main;
    void *cubin_main = read_file("main.cubin", &sizeof_cubin_main);

    frep_t *tree = fBoxCheap(1.0f, 0.5f, 0.25f);

    size_t sizeof_cubin_tree;
    void *cubin_tree = frep_compile_to_sass(tree, &sizeof_cubin_tree);

    CUmodule module = 0;
    link_sass(&module, cubin_main, sizeof_cubin_main, cubin_tree, sizeof_cubin_tree);

    CUfunction kernel;
    cudaCheckError(cuModuleGetFunction(&kernel, module, "main")); assert(kernel);

    //
    // finally we run the thing to make sure that it actually works.
    //
    int N = 32;
    size_t sizeof_input = 4*N*sizeof(float);
    size_t sizeof_output = N*sizeof(float);
    float *input = (float*)malloc(sizeof_input);
    float *output = (float*)malloc(sizeof_output);

    for (int i = 0; i < N; i++)
    {
        input[4*i + 0] = 1.0f;
        input[4*i + 1] = 0.0f;
        input[4*i + 2] = 0.0f;
        input[4*i + 3] = 0.0f;
    }

    int num_blocks = 8;
    int num_threads = 4;
    int shared_memory_bytes = 1024;
    CUdeviceptr dev_input;
    CUdeviceptr dev_output;
    cudaCheckError(cuMemAlloc(&dev_input, sizeof_input)); assert(dev_input);
    cudaCheckError(cuMemAlloc(&dev_output, sizeof_output)); assert(dev_output);
    cudaCheckError(cuMemcpyHtoD(dev_input, input, sizeof_input));
    uint64_t param0 = (uint64_t)(dev_input);
    uint64_t param1 = (uint64_t)(dev_output);
    void *kernel_params[] = { (void*)&param0, (void*)&param1 };
    cuLaunchKernel(kernel, num_blocks,1,1, num_threads,1,1, shared_memory_bytes, NULL, kernel_params, NULL);
    cudaCheckError(cuCtxSynchronize());
    cudaCheckError(cuMemcpyDtoH(output, dev_output, sizeof_output));
    cudaCheckError(cuMemFree(dev_output));
    cudaCheckError(cuMemFree(dev_input));

    cudaCheckError(cuModuleUnload(module));

    printf("output:\n");
    for (int i = 0; i < N; i++)
        printf("%f ", output[i]);

    return 0;
}

void link_sass(CUmodule *module,
               void *cubin1, size_t sizeof_cubin1,
               void *cubin2, size_t sizeof_cubin2)
{
    enum { num_options = 6 };
    CUjit_option options[num_options];
    void *option_values[num_options];
    char error_log[8192];
    char info_log[8192];
    options[0] = CU_JIT_INFO_LOG_BUFFER;             option_values[0] = (void *) info_log;
    options[1] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;  option_values[1] = (void *) (long)sizeof(info_log);
    options[2] = CU_JIT_ERROR_LOG_BUFFER;            option_values[2] = (void *) error_log;
    options[3] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES; option_values[3] = (void *) (long)sizeof(error_log);
    options[4] = CU_JIT_LOG_VERBOSE;                 option_values[4] = (void *) 1;
    options[5] = CU_JIT_TARGET;                      option_values[5] = (void *) CU_TARGET_COMPUTE_60;
    CUlinkState link_state;
    cudaCheckError(cuLinkCreate(num_options, options, option_values, &link_state));

    if (CUDA_SUCCESS != cuLinkAddData(link_state, CU_JIT_INPUT_CUBIN,
        (void *)cubin_main, sizeof_cubin_main, 0,0,0,0))
        fprintf(stderr, "nvlink error:\n%s\n", error_log);

    if (CUDA_SUCCESS != cuLinkAddData(link_state, CU_JIT_INPUT_CUBIN,
        (void *)cubin_tree, sizeof_cubin_tree, 0,0,0,0))
        fprintf(stderr, "nvlink error:\n%s\n", error_log);

    void *cubin;
    size_t cubin_size;
    cudaCheckError(cuLinkComplete(link_state, &cubin, &cubin_size));
    cudaCheckError(cuModuleLoadData(module, cubin)); assert(module);
    cudaCheckError(cuLinkDestroy(link_state));
}
