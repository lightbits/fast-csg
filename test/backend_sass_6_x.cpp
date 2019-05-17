// Compile .cu source file to .ptx
//  nvcc -ptx source.cu --gpu-architecture=sm_60 --use_fast_math
// Compile .ptx to cubin
//  ptxas --opt-level 1 --gpu-name sm_60 source.ptx --output-file source.cubin
// Disassemble cubin
//  nvdisasm -hex source.cubin > source.nvdisasm

// ptxas --opt-level 3 --gpu-name sm_60 test4.ptx --output-file disasm_samples/test4.cubin && ptxas --opt-level 3 --gpu-name sm_60 test5.ptx --output-file disasm_samples/test5.cubin && ptxas --opt-level 3 --gpu-name sm_60 test6.ptx --output-file disasm_samples/test6.cubin && ptxas --opt-level 3 --gpu-name sm_60 test7.ptx --output-file disasm_samples/test7.cubin
// nvdisasm -hex disasm_samples/test4.cubin > disasm_samples/test4.nvdisasm && nvdisasm -hex disasm_samples/test5.cubin > disasm_samples/test5.nvdisasm && nvdisasm -hex disasm_samples/test6.cubin > disasm_samples/test6.nvdisasm && nvdisasm -hex disasm_samples/test7.cubin > disasm_samples/test7.nvdisasm

#include <iostream>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "cuda_error.h"
#define ENABLE_TIMING
#include "profiler.h"
#include "point_data/points3.h"

#define CUBIN_FILENAME         "../codegen/disasm_samples/test5.cubin"
#define PATCHED_CUBIN_FILENAME "../codegen/disasm_samples/test5.patched.cubin"
#define CUBIN_ENTRYPOINT       "test"

#define Assert(Expression) if (!(Expression)) { fprintf(stderr, "Assertation failed \"%s\" in file %s line %d\n", #Expression, __FILE__, __LINE__); cudaDeviceReset(); exit(EXIT_FAILURE); }

CUmodule LoadCubinFromFile(const char *filename)
{
    char *cubin;
    {
        FILE *f = fopen(filename, "rb");
        Assert(f);
        fseek(f, 0, SEEK_END);
        long size = ftell(f);
        rewind(f);
        cubin = new char[size + 1];
        int ok = fread(cubin, 1, size, f);
        Assert(ok);
        cubin[size] = 0;
        fclose(f);
    }
    CUmodule module;
    cudaCheckError(cuModuleLoadData(&module, cubin)); Assert(module);
    // todo: can we free cubin here? see cuModuleLoadData, cuLink* in Driver API
    return module;
}

void InitializeCUDA()
{
    int device_id = 0; // always choose device index 0... change this if you have multiple devices
    int device_count;
    cudaCheckError(cudaGetDeviceCount(&device_count));
    Assert(device_count >= 1 && "No CUDA capable devices found");

    cudaDeviceProp prop;
    cudaCheckError(cudaGetDeviceProperties(&prop, device_id));
    Assert(prop.computeMode != cudaComputeModeProhibited && "Device is running in Compute Mode Prohibited, no threads can use cudaSetDevice()");
    Assert(prop.major >= 1 && "GPU device does not support CUDA");
    cudaCheckError(cudaSetDevice(device_id));
    printf("Using CUDA Device %d: \"%s\n", device_id, prop.name);
}

int main(int argc, char **argv)
{
    setenv("CUDA_CACHE_DISABLE", "1", 1);
    InitializeCUDA();

    enum { num_threads = 32 };
    enum { num_blocks  = NUM_POINTS/num_threads };
    enum { num_output = 1 };
    enum { sizeof_input = NUM_POINTS*4*sizeof(float) };
    enum { sizeof_output = NUM_POINTS*sizeof(float)*num_output };

    const float *input = points;
    float *output = (float*)malloc(sizeof_output); Assert(output);
    void *dev_input = NULL;
    void *dev_output = NULL;
    cudaCheckError(cudaMalloc(&dev_input, sizeof_input)); Assert(dev_input);
    cudaCheckError(cudaMalloc(&dev_output, sizeof_output)); Assert(dev_output);
    cudaCheckError(cudaMemcpy(dev_input, input, sizeof_input, cudaMemcpyHostToDevice));

    for (int which = 0; which < 2; which++)
    {
        CUmodule module;
        if (which == 0)
        {
            module = LoadCubinFromFile(CUBIN_FILENAME);
            printf("Running original binary...\n");
        }
        else
        {
            TIMING("cubin");
            module = LoadCubinFromFile(PATCHED_CUBIN_FILENAME);
            TIMING("cubin");
            printf("Running patched binary...\n");
        }
        CUfunction kernel = 0;
        cudaCheckError(cuModuleGetFunction(&kernel, module, CUBIN_ENTRYPOINT));

        uint64_t param0 = (uint64_t)(dev_input);
        uint64_t param1 = (uint64_t)(dev_output);
        void *kernel_params[] = { (void*)&param0, (void*)&param1 };
        int shared_memory_bytes = 1024;
        cuLaunchKernel(kernel, num_blocks,1,1, num_threads,1,1, shared_memory_bytes, NULL, kernel_params, NULL);
        cudaCheckError(cuCtxSynchronize());
        cudaCheckError(cudaMemcpy(output, dev_output, sizeof_output, cudaMemcpyDeviceToHost));
        cudaCheckError(cuModuleUnload(module));

        for (int j = 0; j < NUM_POINTS && j < 16; j++)
        {
            float x = input[4*j+0];
            float y = input[4*j+1];
            float z = input[4*j+2];
            #if 0
            float r0 = x*y;
            float r1 = y*z;
            float r3 = z*z;
            float r4 = r0*r1 + r3;
            #endif
            #if 0
            float r5 = y*y;
            float r4 = r5 + z;
            #endif
            #if 1
            float r4 = 2*sqrtf(y*y);
            #endif
            printf("%d: %f %f\n", j, output[j], r4);
        }
    }

    cudaCheckError(cudaFree(dev_output));
    cudaCheckError(cudaFree(dev_input));
    free(output);

    TIMING_SUMMARY();

    return 0;
}
