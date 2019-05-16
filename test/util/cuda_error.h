static const char *cudaErrorToString(CUresult error)
{
    switch (error)
    {
        case CUDA_SUCCESS:
            return "CUDA_SUCCESS";

        case CUDA_ERROR_INVALID_VALUE:
            return "CUDA_ERROR_INVALID_VALUE";

        case CUDA_ERROR_OUT_OF_MEMORY:
            return "CUDA_ERROR_OUT_OF_MEMORY";

        case CUDA_ERROR_NOT_INITIALIZED:
            return "CUDA_ERROR_NOT_INITIALIZED";

        case CUDA_ERROR_DEINITIALIZED:
            return "CUDA_ERROR_DEINITIALIZED";

        case CUDA_ERROR_PROFILER_DISABLED:
            return "CUDA_ERROR_PROFILER_DISABLED";

        case CUDA_ERROR_PROFILER_NOT_INITIALIZED:
            return "CUDA_ERROR_PROFILER_NOT_INITIALIZED";

        case CUDA_ERROR_PROFILER_ALREADY_STARTED:
            return "CUDA_ERROR_PROFILER_ALREADY_STARTED";

        case CUDA_ERROR_PROFILER_ALREADY_STOPPED:
            return "CUDA_ERROR_PROFILER_ALREADY_STOPPED";

        case CUDA_ERROR_NO_DEVICE:
            return "CUDA_ERROR_NO_DEVICE";

        case CUDA_ERROR_INVALID_DEVICE:
            return "CUDA_ERROR_INVALID_DEVICE";

        case CUDA_ERROR_INVALID_IMAGE:
            return "CUDA_ERROR_INVALID_IMAGE";

        case CUDA_ERROR_INVALID_CONTEXT:
            return "CUDA_ERROR_INVALID_CONTEXT";

        case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:
            return "CUDA_ERROR_CONTEXT_ALREADY_CURRENT";

        case CUDA_ERROR_MAP_FAILED:
            return "CUDA_ERROR_MAP_FAILED";

        case CUDA_ERROR_UNMAP_FAILED:
            return "CUDA_ERROR_UNMAP_FAILED";

        case CUDA_ERROR_ARRAY_IS_MAPPED:
            return "CUDA_ERROR_ARRAY_IS_MAPPED";

        case CUDA_ERROR_ALREADY_MAPPED:
            return "CUDA_ERROR_ALREADY_MAPPED";

        case CUDA_ERROR_NO_BINARY_FOR_GPU:
            return "CUDA_ERROR_NO_BINARY_FOR_GPU";

        case CUDA_ERROR_ALREADY_ACQUIRED:
            return "CUDA_ERROR_ALREADY_ACQUIRED";

        case CUDA_ERROR_NOT_MAPPED:
            return "CUDA_ERROR_NOT_MAPPED";

        case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:
            return "CUDA_ERROR_NOT_MAPPED_AS_ARRAY";

        case CUDA_ERROR_NOT_MAPPED_AS_POINTER:
            return "CUDA_ERROR_NOT_MAPPED_AS_POINTER";

        case CUDA_ERROR_ECC_UNCORRECTABLE:
            return "CUDA_ERROR_ECC_UNCORRECTABLE";

        case CUDA_ERROR_UNSUPPORTED_LIMIT:
            return "CUDA_ERROR_UNSUPPORTED_LIMIT";

        case CUDA_ERROR_CONTEXT_ALREADY_IN_USE:
            return "CUDA_ERROR_CONTEXT_ALREADY_IN_USE";

        case CUDA_ERROR_PEER_ACCESS_UNSUPPORTED:
            return "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED";

        case CUDA_ERROR_INVALID_PTX:
            return "CUDA_ERROR_INVALID_PTX";

        case CUDA_ERROR_INVALID_GRAPHICS_CONTEXT:
            return "CUDA_ERROR_INVALID_GRAPHICS_CONTEXT";

        case CUDA_ERROR_NVLINK_UNCORRECTABLE:
            return "CUDA_ERROR_NVLINK_UNCORRECTABLE";

        case CUDA_ERROR_JIT_COMPILER_NOT_FOUND:
            return "CUDA_ERROR_JIT_COMPILER_NOT_FOUND";

        case CUDA_ERROR_INVALID_SOURCE:
            return "CUDA_ERROR_INVALID_SOURCE";

        case CUDA_ERROR_FILE_NOT_FOUND:
            return "CUDA_ERROR_FILE_NOT_FOUND";

        case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
            return "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND";

        case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
            return "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED";

        case CUDA_ERROR_OPERATING_SYSTEM:
            return "CUDA_ERROR_OPERATING_SYSTEM";

        case CUDA_ERROR_INVALID_HANDLE:
            return "CUDA_ERROR_INVALID_HANDLE";

        case CUDA_ERROR_NOT_FOUND:
            return "CUDA_ERROR_NOT_FOUND";

        case CUDA_ERROR_NOT_READY:
            return "CUDA_ERROR_NOT_READY";

        case CUDA_ERROR_ILLEGAL_ADDRESS:
            return "CUDA_ERROR_ILLEGAL_ADDRESS";

        case CUDA_ERROR_LAUNCH_FAILED:
            return "CUDA_ERROR_LAUNCH_FAILED";

        case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
            return "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES";

        case CUDA_ERROR_LAUNCH_TIMEOUT:
            return "CUDA_ERROR_LAUNCH_TIMEOUT";

        case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
            return "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING";

        case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
            return "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED";

        case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:
            return "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED";

        case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE:
            return "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE";

        case CUDA_ERROR_CONTEXT_IS_DESTROYED:
            return "CUDA_ERROR_CONTEXT_IS_DESTROYED";

        case CUDA_ERROR_ASSERT:
            return "CUDA_ERROR_ASSERT";

        case CUDA_ERROR_TOO_MANY_PEERS:
            return "CUDA_ERROR_TOO_MANY_PEERS";

        case CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
            return "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED";

        case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED:
            return "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED";

        case CUDA_ERROR_HARDWARE_STACK_ERROR:
            return "CUDA_ERROR_HARDWARE_STACK_ERROR";

        case CUDA_ERROR_ILLEGAL_INSTRUCTION:
            return "CUDA_ERROR_ILLEGAL_INSTRUCTION";

        case CUDA_ERROR_MISALIGNED_ADDRESS:
            return "CUDA_ERROR_MISALIGNED_ADDRESS";

        case CUDA_ERROR_INVALID_ADDRESS_SPACE:
            return "CUDA_ERROR_INVALID_ADDRESS_SPACE";

        case CUDA_ERROR_INVALID_PC:
            return "CUDA_ERROR_INVALID_PC";

        case CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE:
            return "CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE";

        case CUDA_ERROR_NOT_PERMITTED:
            return "CUDA_ERROR_NOT_PERMITTED";

        case CUDA_ERROR_NOT_SUPPORTED:
            return "CUDA_ERROR_NOT_SUPPORTED";

        case CUDA_ERROR_UNKNOWN:
            return "CUDA_ERROR_UNKNOWN";
    }

    return "<unknown>";
}

static const char *cudaErrorToString(cudaError_t error)
{
    switch (error)
    {
        case cudaSuccess:
            return "cudaSuccess";

        case cudaErrorMissingConfiguration:
            return "cudaErrorMissingConfiguration";

        case cudaErrorMemoryAllocation:
            return "cudaErrorMemoryAllocation";

        case cudaErrorInitializationError:
            return "cudaErrorInitializationError";

        case cudaErrorLaunchFailure:
            return "cudaErrorLaunchFailure";

        case cudaErrorPriorLaunchFailure:
            return "cudaErrorPriorLaunchFailure";

        case cudaErrorLaunchTimeout:
            return "cudaErrorLaunchTimeout";

        case cudaErrorLaunchOutOfResources:
            return "cudaErrorLaunchOutOfResources";

        case cudaErrorInvalidDeviceFunction:
            return "cudaErrorInvalidDeviceFunction";

        case cudaErrorInvalidConfiguration:
            return "cudaErrorInvalidConfiguration";

        case cudaErrorInvalidDevice:
            return "cudaErrorInvalidDevice";

        case cudaErrorInvalidValue:
            return "cudaErrorInvalidValue";

        case cudaErrorInvalidPitchValue:
            return "cudaErrorInvalidPitchValue";

        case cudaErrorInvalidSymbol:
            return "cudaErrorInvalidSymbol";

        case cudaErrorMapBufferObjectFailed:
            return "cudaErrorMapBufferObjectFailed";

        case cudaErrorUnmapBufferObjectFailed:
            return "cudaErrorUnmapBufferObjectFailed";

        case cudaErrorInvalidHostPointer:
            return "cudaErrorInvalidHostPointer";

        case cudaErrorInvalidDevicePointer:
            return "cudaErrorInvalidDevicePointer";

        case cudaErrorInvalidTexture:
            return "cudaErrorInvalidTexture";

        case cudaErrorInvalidTextureBinding:
            return "cudaErrorInvalidTextureBinding";

        case cudaErrorInvalidChannelDescriptor:
            return "cudaErrorInvalidChannelDescriptor";

        case cudaErrorInvalidMemcpyDirection:
            return "cudaErrorInvalidMemcpyDirection";

        case cudaErrorAddressOfConstant:
            return "cudaErrorAddressOfConstant";

        case cudaErrorTextureFetchFailed:
            return "cudaErrorTextureFetchFailed";

        case cudaErrorTextureNotBound:
            return "cudaErrorTextureNotBound";

        case cudaErrorSynchronizationError:
            return "cudaErrorSynchronizationError";

        case cudaErrorInvalidFilterSetting:
            return "cudaErrorInvalidFilterSetting";

        case cudaErrorInvalidNormSetting:
            return "cudaErrorInvalidNormSetting";

        case cudaErrorMixedDeviceExecution:
            return "cudaErrorMixedDeviceExecution";

        case cudaErrorCudartUnloading:
            return "cudaErrorCudartUnloading";

        case cudaErrorUnknown:
            return "cudaErrorUnknown";

        case cudaErrorNotYetImplemented:
            return "cudaErrorNotYetImplemented";

        case cudaErrorMemoryValueTooLarge:
            return "cudaErrorMemoryValueTooLarge";

        case cudaErrorInvalidResourceHandle:
            return "cudaErrorInvalidResourceHandle";

        case cudaErrorNotReady:
            return "cudaErrorNotReady";

        case cudaErrorInsufficientDriver:
            return "cudaErrorInsufficientDriver";

        case cudaErrorSetOnActiveProcess:
            return "cudaErrorSetOnActiveProcess";

        case cudaErrorInvalidSurface:
            return "cudaErrorInvalidSurface";

        case cudaErrorNoDevice:
            return "cudaErrorNoDevice";

        case cudaErrorECCUncorrectable:
            return "cudaErrorECCUncorrectable";

        case cudaErrorSharedObjectSymbolNotFound:
            return "cudaErrorSharedObjectSymbolNotFound";

        case cudaErrorSharedObjectInitFailed:
            return "cudaErrorSharedObjectInitFailed";

        case cudaErrorUnsupportedLimit:
            return "cudaErrorUnsupportedLimit";

        case cudaErrorDuplicateVariableName:
            return "cudaErrorDuplicateVariableName";

        case cudaErrorDuplicateTextureName:
            return "cudaErrorDuplicateTextureName";

        case cudaErrorDuplicateSurfaceName:
            return "cudaErrorDuplicateSurfaceName";

        case cudaErrorDevicesUnavailable:
            return "cudaErrorDevicesUnavailable";

        case cudaErrorInvalidKernelImage:
            return "cudaErrorInvalidKernelImage";

        case cudaErrorNoKernelImageForDevice:
            return "cudaErrorNoKernelImageForDevice";

        case cudaErrorIncompatibleDriverContext:
            return "cudaErrorIncompatibleDriverContext";

        case cudaErrorPeerAccessAlreadyEnabled:
            return "cudaErrorPeerAccessAlreadyEnabled";

        case cudaErrorPeerAccessNotEnabled:
            return "cudaErrorPeerAccessNotEnabled";

        case cudaErrorDeviceAlreadyInUse:
            return "cudaErrorDeviceAlreadyInUse";

        case cudaErrorProfilerDisabled:
            return "cudaErrorProfilerDisabled";

        case cudaErrorProfilerNotInitialized:
            return "cudaErrorProfilerNotInitialized";

        case cudaErrorProfilerAlreadyStarted:
            return "cudaErrorProfilerAlreadyStarted";

        case cudaErrorProfilerAlreadyStopped:
            return "cudaErrorProfilerAlreadyStopped";

        /* Since CUDA 4.0*/
        case cudaErrorAssert:
            return "cudaErrorAssert";

        case cudaErrorTooManyPeers:
            return "cudaErrorTooManyPeers";

        case cudaErrorHostMemoryAlreadyRegistered:
            return "cudaErrorHostMemoryAlreadyRegistered";

        case cudaErrorHostMemoryNotRegistered:
            return "cudaErrorHostMemoryNotRegistered";

        /* Since CUDA 5.0 */
        case cudaErrorOperatingSystem:
            return "cudaErrorOperatingSystem";

        case cudaErrorPeerAccessUnsupported:
            return "cudaErrorPeerAccessUnsupported";

        case cudaErrorLaunchMaxDepthExceeded:
            return "cudaErrorLaunchMaxDepthExceeded";

        case cudaErrorLaunchFileScopedTex:
            return "cudaErrorLaunchFileScopedTex";

        case cudaErrorLaunchFileScopedSurf:
            return "cudaErrorLaunchFileScopedSurf";

        case cudaErrorSyncDepthExceeded:
            return "cudaErrorSyncDepthExceeded";

        case cudaErrorLaunchPendingCountExceeded:
            return "cudaErrorLaunchPendingCountExceeded";

        case cudaErrorNotPermitted:
            return "cudaErrorNotPermitted";

        case cudaErrorNotSupported:
            return "cudaErrorNotSupported";

        /* Since CUDA 6.0 */
        case cudaErrorHardwareStackError:
            return "cudaErrorHardwareStackError";

        case cudaErrorIllegalInstruction:
            return "cudaErrorIllegalInstruction";

        case cudaErrorMisalignedAddress:
            return "cudaErrorMisalignedAddress";

        case cudaErrorInvalidAddressSpace:
            return "cudaErrorInvalidAddressSpace";

        case cudaErrorInvalidPc:
            return "cudaErrorInvalidPc";

        case cudaErrorIllegalAddress:
            return "cudaErrorIllegalAddress";

        /* Since CUDA 6.5*/
        case cudaErrorInvalidPtx:
            return "cudaErrorInvalidPtx";

        case cudaErrorInvalidGraphicsContext:
            return "cudaErrorInvalidGraphicsContext";

        case cudaErrorStartupFailure:
            return "cudaErrorStartupFailure";

        case cudaErrorApiFailureBase:
            return "cudaErrorApiFailureBase";

        /* Since CUDA 8.0*/
        case cudaErrorNvlinkUncorrectable :
            return "cudaErrorNvlinkUncorrectable";

        /* Since CUDA 8.5*/
        case cudaErrorJitCompilerNotFound :
            return "cudaErrorJitCompilerNotFound";

        /* Since CUDA 9.0*/
        case cudaErrorCooperativeLaunchTooLarge :
            return "cudaErrorCooperativeLaunchTooLarge";

    }

    return "<unknown>";
}

template< typename T >
void _cudaCheckError(T result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<unsigned int>(result), cudaErrorToString(result), func);
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}
#define cudaCheckError(val) _cudaCheckError ( (val), #val, __FILE__, __LINE__ )
