// RunPTX and RunCubin launch the kernel contained in the provided PTX or Cubin
// program, copying the input array to the device and copying the output back to
// host memory (stored in the provided output pointer).

// RunPTX and RunCubin assume that the program you provide has a single entry point
// that takes two parameters, i.e.
//   .func <entry_name>(.param.u64 param0, .param.u64 param1)
//   {
//       ...
//   }
// When the kernel is invoked, we pass into its parameters the following:
//   param0 = (uint64_t)input
//   param1 = (uint64_t)output
// i.e. the first parameter is the address of the kernel INPUT and the
// second parameter is the address of the kernel OUTPUT. I hardcode
// assume 64-bit addressing for now.

// Note: cudaMalloc in RunPTX and RunCubin implicitly initializes the CUDA context.
// See cuda/samples/vectorAdd_nvrtc and common/inc/nvrtc_helper.h for how to use Driver API to explicitly initialize
#pragma once
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_error.h"

CUmodule _LoadPTXFromString(const char *ptx_source, size_t ptx_source_length, int jit_optimization_level);
void _InitializeCUDA();

void RunPTX(void *input, size_t sizeof_input,
            void *output, size_t sizeof_output,
            const char *ptx_source, size_t ptx_source_length, const char *entry_name,
            int num_blocks, int num_threads, int shared_memory_bytes=1024,
            int jit_optimization_level=1 /*allowed values = 0,1,2,3,4*/)
{
    _InitializeCUDA();
    CUdeviceptr dev_input;
    CUdeviceptr dev_output;
    cudaCheckError(cuMemAlloc(&dev_input, sizeof_input)); assert(dev_input);
    cudaCheckError(cuMemAlloc(&dev_output, sizeof_output)); assert(dev_output);
    cudaCheckError(cuMemcpyHtoD(dev_input, input, sizeof_input));
    CUmodule module = _LoadPTXFromString(ptx_source, ptx_source_length, jit_optimization_level);
    CUfunction kernel = 0;
    cudaCheckError(cuModuleGetFunction(&kernel, module, entry_name));
    uint64_t param0 = (uint64_t)(dev_input);
    uint64_t param1 = (uint64_t)(dev_output);
    void *kernel_params[] = { (void*)&param0, (void*)&param1 };
    cuLaunchKernel(kernel, num_blocks,1,1, num_threads,1,1, shared_memory_bytes, NULL, kernel_params, NULL);
    cudaCheckError(cuCtxSynchronize());
    cudaCheckError(cuMemcpyDtoH(output, dev_output, sizeof_output));
    cudaCheckError(cuMemFree(dev_output));
    cudaCheckError(cuMemFree(dev_input));
    cudaCheckError(cuModuleUnload(module));
}

#if 0
void RunCubin(void *input, size_t sizeof_input,
              void *output, size_t sizeof_output,
              const void *cubin, const char *entry_name,
              int num_blocks, int num_threads, int shared_memory_bytes=1024)
{
    _InitializeCUDA();
    void *dev_input = NULL;
    void *dev_output = NULL;
    CUmodule module;
    CUfunction kernel;
    cudaCheckError(cudaMalloc(&dev_input, sizeof_input)); assert(dev_input);
    cudaCheckError(cudaMalloc(&dev_output, sizeof_output)); assert(dev_output);
    cudaCheckError(cudaMemcpy(dev_input, input, sizeof_input, cudaMemcpyHostToDevice));
    cudaCheckError(cuModuleLoadData(&module, cubin)); assert(module);
    cudaCheckError(cuModuleGetFunction(&kernel, module, entry_name)); assert(kernel);
    uint64_t param0 = (uint64_t)(dev_input);
    uint64_t param1 = (uint64_t)(dev_output);
    void *kernel_params[] = { (void*)&param0, (void*)&param1 };
    cuLaunchKernel(kernel, num_blocks,1,1, num_threads,1,1, shared_memory_bytes, NULL, kernel_params, NULL);
    cudaCheckError(cuCtxSynchronize());
    cudaCheckError(cudaMemcpy(output, dev_output, sizeof_output, cudaMemcpyDeviceToHost));
    cudaCheckError(cudaFree(dev_output));
    cudaCheckError(cudaFree(dev_input));
    cudaCheckError(cuModuleUnload(module));
}
#endif

CUmodule _LoadPTXFromString(const char *ptx_source, size_t ptx_source_length, int jit_optimization_level)
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

void _InitializeCUDA()
{
    static bool has_init = false;
    if (has_init)
        return;
    has_init = true;

    // disable CUDA from caching SASS programs
    setenv("CUDA_CACHE_DISABLE", "1", 1);

    CUcontext context;
    CUdevice device;
    cudaCheckError(cuInit(0));
    cudaCheckError(cuDeviceGet(&device, 0));
    cudaCheckError(cuCtxCreate(&context, 0, device));

    char name[256];
    int major = 0, minor = 0;
    int compute_mode = -1;

    cudaCheckError(cuDeviceGetName(name, 100, device));
    cudaCheckError(cuDeviceGetAttribute(
        &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    cudaCheckError(cuDeviceGetAttribute(
        &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
    cudaCheckError(cuDeviceGetAttribute(
        &compute_mode, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, device));
    assert(compute_mode != CU_COMPUTEMODE_PROHIBITED && "Device is running in Compute Mode Prohibited");
    printf("Using CUDA device %s: Compute SM %d.%d\n", name, major, minor);
}
