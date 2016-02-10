#include <sstream>

#include "nn/Gpu.h"
#include "common/Common.h"

using namespace std;

namespace nn {

#define C(id, program, kernel) program
static string program_name_for_kernel[] = {
    #include "nn/KernelList.h"
};
#undef C

#define C(id, program, kernel) kernel
static string name_for_kernel[] = {
    #include "nn/KernelList.h"
};
#undef C

KernelManager::~KernelManager()
{
    // Free all kernels
    for (size_t i = 0; i < kNumKernels; i++) {
        delete kernels_[i];
    }

    // Free all programs
    for (auto p : programs_) {
        delete p.second;
    }
}

bool KernelManager::LoadKernels(const string& kernel_directory)
{
    Assert(GPUContext::device);

    string compile_options = "-I " + kernel_directory;
    kernel_directory_ = kernel_directory;

    for (size_t i = 0; i < kNumKernels; i++) {
        string program_name = program_name_for_kernel[i];
        if (programs_.count(program_name) == 0) {
            programs_[program_name] = GPUContext::device->CreateProgramFromFile(kernel_directory + program_name + ".cl", compile_options).release();
            if (!programs_[program_name])
                return false;
        }
        kernels_[i] = programs_[program_name]->CreateKernel(name_for_kernel[i]).release();
        if (!kernels_[i])
            return false;
    }

    return true;
}

ocl::Kernel* KernelManager::convolution_kernel(size_t kernel_width, size_t kernel_height)
{
    Assert(GPUContext::device);

    size_t halfwidth = kernel_width/2, halfheight = kernel_height/2;
    if (!convolution_kernels_[halfwidth][halfheight]) {
        // These must be in sync with the .cl source.
        constexpr size_t kTileWidth = 16, kTileHeight = 16;

        stringstream compile_options;
        compile_options << "-I " + kernel_directory_;
        compile_options << " -D KERNEL_WIDTH=" << kernel_width;
        compile_options << " -D KERNEL_HEIGHT=" << kernel_height;

        // Compute the halo lookup table. The lookup table assigns each thread a set of halo pixels to load. See kernels/Convolution.cl
        stringstream lookup_table_x, lookup_table_y;
        string separator = "";
        for (size_t y = 0; y < kTileHeight + 2 * halfheight; y++) {
            for (size_t x = 0; x < kTileWidth + 2 * halfwidth; x++) {
                if (x < halfwidth || x >= kTileWidth + halfwidth || y < halfheight || y >= kTileHeight + halfheight) {
                    lookup_table_x << separator << x;
                    lookup_table_y << separator << y;
                    separator = ",";
                }
            }
        }

        compile_options << " -D LOOKUP_TABLE_X={" << lookup_table_x.str() << "}";
        compile_options << " -D LOOKUP_TABLE_Y={" << lookup_table_y.str() << "}";

        auto program = GPUContext::device->CreateProgramFromFile(kernel_directory_ + "Convolution.cl", compile_options.str()).release();
        convolution_kernels_[halfwidth][halfheight] = program->CreateKernel("Convolution2D").release();
        cross_correlation_kernels_[halfwidth][halfheight] = program->CreateKernel("CrossCorrelation2D").release();
        convolution_gradient_kernels_[halfwidth][halfheight] = program->CreateKernel("Convolution2DGradients").release();
    }

    return convolution_kernels_[halfwidth][halfheight];
}

ocl::Kernel* KernelManager::cross_correlation_kernel(size_t kernel_width, size_t kernel_height)
{
    Assert(GPUContext::device);

    size_t halfwidth = kernel_width/2, halfheight = kernel_height/2;
    if (!cross_correlation_kernels_[halfwidth][halfheight]) {
        // Load convoltion and cross-correlation kernels
        convolution_kernel(kernel_width, kernel_height);
    }

    return cross_correlation_kernels_[halfwidth][halfheight];
}

ocl::Kernel* KernelManager::convolution_gradient_kernel(size_t kernel_width, size_t kernel_height)
{
    Assert(GPUContext::device);

    size_t halfwidth = kernel_width/2, halfheight = kernel_height/2;
    if (!convolution_gradient_kernels_[halfwidth][halfheight]) {
        // Load convoltion and cross-correlation kernels
        convolution_kernel(kernel_width, kernel_height);
    }

    return convolution_gradient_kernels_[halfwidth][halfheight];
}

ocl::Device* GPUContext::device = nullptr;;
KernelManager GPUContext::kernel_manager;

bool GPUContext::Init(ocl::Device* device, const string& kernel_directory)
{
    Assert(!GPUContext::device);
    GPUContext::device = device;
    return kernel_manager.LoadKernels(kernel_directory);
}

}       // namespace nn
