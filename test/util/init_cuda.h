#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "cuda_error.h"
void init_cuda()
{
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
    cudaCheckError(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    cudaCheckError(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
    cudaCheckError(cuDeviceGetAttribute(&compute_mode, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, device));
    assert(compute_mode != CU_COMPUTEMODE_PROHIBITED && "Device is running in Compute Mode Prohibited");
    printf("Using CUDA device %s: Compute SM %d.%d\n", name, major, minor);
}
