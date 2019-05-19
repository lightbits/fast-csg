// This file tests the use of seperate compilation to link together
// pre-existing Cubin files. This is useful because it means we can
// use the CUDA Driver API to generate an executable Cubin from the
// SASS backend and an externally provided Cubin containing a "main"
// entrypoint.

// Compilation instructions (Linux, g++):
// g++ -std=c++11 linker.cpp -I/usr/local/cuda-10.1/include -lcuda
#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include "util/cuda_error.h"
#include "util/init_cuda.h"
#include "util/read_file.h"
#define ENABLE_TIMING
#include "util/profiler.h"

int main() {
    init_cuda();

    system("/usr/local/cuda-10.1/bin/ptxas --opt-level 1 --compile-only --gpu-name sm_60 test1.ptx --output-file test1.cubin");
    system("/usr/local/cuda-10.1/bin/ptxas --opt-level 1 --compile-only --gpu-name sm_60 test2.ptx --output-file test2.cubin");

    int sizeof_cubin1 = 0;
    void *cubin1 = (void*)read_file("test1.cubin", &sizeof_cubin1);
    assert(cubin1);
    int sizeof_cubin2 = 0;
    void *cubin2 = (void*)read_file("test2.cubin", &sizeof_cubin2);
    assert(cubin2);

    CUfunction kernel;
    CUmodule module;
    const char *entry_name = "main";
    for (int i = 0; i < 100; i++)
    {
        TIMING("linker");
        void *cubin; size_t cubin_size;
        CUlinkState link_state;
        enum { num_options = 6 };
        CUjit_option options[num_options];
        void *option_values[num_options];
        char error_log[8192], info_log[8192];

        // see CUDA Driver API manual for these options (look up cuLinkCreate)
        options[0] = CU_JIT_INFO_LOG_BUFFER;             option_values[0] = (void *) info_log;
        options[1] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;  option_values[1] = (void *) (long)sizeof(info_log);
        options[2] = CU_JIT_ERROR_LOG_BUFFER;            option_values[2] = (void *) error_log;
        options[3] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES; option_values[3] = (void *) (long)sizeof(error_log);
        options[4] = CU_JIT_LOG_VERBOSE;                 option_values[4] = (void *) 1;
        options[5] = CU_JIT_TARGET;                      option_values[5] = (void *) CU_TARGET_COMPUTE_60;
        cudaCheckError(cuLinkCreate(num_options, options, option_values, &link_state));

        {
            int err = cuLinkAddData(link_state, CU_JIT_INPUT_CUBIN, (void *)cubin1, sizeof_cubin1, 0, 0, 0, 0);
            if (err != CUDA_SUCCESS)
                fprintf(stderr, "PTX Linker Error:\n%s\n", error_log);
        }
        {
            int err = cuLinkAddData(link_state, CU_JIT_INPUT_CUBIN, (void *)cubin2, sizeof_cubin2, 0, 0, 0, 0);
            if (err != CUDA_SUCCESS)
                fprintf(stderr, "PTX Linker Error:\n%s\n", error_log);
        }
        cudaCheckError(cuLinkComplete(link_state, &cubin, &cubin_size));
        cudaCheckError(cuModuleLoadData(&module, cubin)); assert(module);
        cudaCheckError(cuLinkDestroy(link_state));
        cudaCheckError(cuModuleGetFunction(&kernel, module, entry_name)); assert(kernel);
        TIMING("linker");
    }
    assert(kernel);
    TIMING_SUMMARY();

    int N = 32;
    size_t sizeof_input = N*sizeof(float);
    size_t sizeof_output = N*sizeof(float);
    float *input = (float*)malloc(sizeof_input);
    float *output = (float*)malloc(sizeof_output);
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
}
