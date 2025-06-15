/*
 * Gpu support.
 *
 * Copyright (c) 2016 Samuel Groß
 */

#ifndef __GPU_H__
#define __GPU_H__

#include <map>

#include "ocl/Device.h"
#include "ocl/Kernel.h"

namespace nn {

// Enum containing a unique identifier for every available OpenCL kernel.
#define C(id, program, kernel) id
enum KernelIDs {
    #include "nn/KernelList.h"
    kNumKernels
};
#undef C

constexpr size_t kMaxConvolutionKernelSize = 11;
constexpr size_t kMaxConvolutionKernelHalfSize = kMaxConvolutionKernelSize / 2 + 1;

// Class to manage OpenCL kernels for the neural networking code.
class KernelManager {
  public:
    KernelManager() { }

    ~KernelManager();

    // Loads all registered kernels (see KernelList.h).
    bool LoadKernels(const std::string& kernel_directory);

    // Returns the kernel with the given ID.
    ocl::Kernel* kernel(KernelIDs id) { return kernels_[id]; }

    // Returns the 2D convolution/cross-correlation kernel for the given kernel size.
    //
    // The OpenCL convolution/cross-correlation kernels are recompiled for every
    // kernel size since these must be fixed at compile time.
    ocl::Kernel* convolution_kernel(size_t kernel_width, size_t kernel_height);
    ocl::Kernel* cross_correlation_kernel(size_t kernel_width, size_t kernel_height);
    ocl::Kernel* convolution_gradient_kernel(size_t kernel_width, size_t kernel_height);

  private:
    ocl::Kernel* kernels_[kNumKernels];
    std::map<std::string, ocl::Program*> programs_;

    // Path to the OpenCL kernel files.
    std::string kernel_directory_;

    // Convolution/cross-correlation kernels.
    // Since widht and height of the convolution kernel must always be odd, we index
    // them by the halfwidth and halfheight of the kernel.
    ocl::Kernel* convolution_kernels_[kMaxConvolutionKernelHalfSize][kMaxConvolutionKernelHalfSize];
    ocl::Kernel* cross_correlation_kernels_[kMaxConvolutionKernelHalfSize][kMaxConvolutionKernelHalfSize];
    ocl::Kernel* convolution_gradient_kernels_[kMaxConvolutionKernelHalfSize][kMaxConvolutionKernelHalfSize];
};


// Class to hold GPU related context for the neural networking code.
//
// Currently, only one OpenCL device is supported, thus all methods of this class are static.
class GPUContext {
  public:
    // Initializes the global GPUContext with the given device.
    // Also loads all registered (see KernelList.h) kernels.
    //
    // Can only be called once per session.
    // Takes ownership of the device pointer.
    static bool Init(ocl::Device* device, const std::string& kernel_directory);

    // Global OpenCL device and kernel manager.
    static ocl::Device* device;
    static KernelManager kernel_manager;
};

}       // namespace nn

#endif
