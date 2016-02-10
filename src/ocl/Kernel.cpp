#include "Kernel.h"

namespace ocl {

Kernel::Kernel(cl_command_queue command_queue, cl_kernel kernel, cl_device_id device) : kernel_(kernel), device_(device), cur_index_(0), command_queue_(command_queue) {
    CL_Check(clRetainCommandQueue(command_queue_));
}

Kernel::~Kernel()
{
    if (kernel_) {
        clReleaseKernel(kernel_);
    }
    if (command_queue_) {
        clReleaseCommandQueue(command_queue_);
    }
}

size_t Kernel::PreferredWorkSizeMultiple() const
{
    size_t size;
    CL_Check(clGetKernelWorkGroupInfo(kernel_, device_, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size), &size, nullptr));
    return size;
}

// We scale down size_t to uint32_t for OpenCL kernels..
template<>
bool Kernel::BindNextArgument<size_t>(size_t size)
{
    Assert(size <= 0xffffffff);

    uint32_t cl_size = size;
    cl_int clErr = clSetKernelArg(kernel_, cur_index_, sizeof(cl_uint), (void *)&cl_size);
    CL_ENSURE_SUCCESS(clErr, "Failed to bind argument for kernel", false);
    cur_index_++;
    return true;
}

template<>
bool Kernel::BindNextArgument<Buffer*>(Buffer* buffer)
{
    cl_mem cl_buffer = buffer->cl_buffer();
    cl_int clErr = clSetKernelArg(kernel_, cur_index_, sizeof(cl_mem), (void *)&cl_buffer);
    CL_ENSURE_SUCCESS(clErr, "Failed to bind buffer argument for kernel", false);
    cur_index_++;
    return true;
}

template<>
bool Kernel::BindNextArgument<LocalMemory>(LocalMemory local_buffer)
{
    cl_int clErr = clSetKernelArg(kernel_, cur_index_, local_buffer.size, NULL);
    CL_ENSURE_SUCCESS(clErr, "Failed to bind argument for kernel", false);
    cur_index_++;
    return true;
}

Kernel::WorkSize Kernel::CalculateLocalWorkSize(WorkSize gws)
{
    // TODO this needs some more love
    WorkSize lws(0);
    if (gws.dimensions == 1)
        lws = WorkSize(256);
    else if (gws.dimensions == 2)
        lws = WorkSize(32, 8);
    else if (gws.dimensions == 3)
        lws = WorkSize(16, 4, 4);
    return lws;
}

bool Kernel::Run(WorkSize gws)
{
    return Run(gws, CalculateLocalWorkSize(gws));
}

bool Kernel::Run(WorkSize gws, WorkSize lws)
{
    Assert(gws.dimensions == lws.dimensions);
    gws = PrepareFinalWorkSize(gws, lws);
    cl_int clErr = clEnqueueNDRangeKernel(command_queue_, kernel_, gws.dimensions, nullptr, gws.values, lws.values, 0, nullptr, nullptr);
    CL_ENSURE_SUCCESS(clErr, "Error executing kernel", false);
    cur_index_ = 0;
    return true;
}

Kernel::WorkSize Kernel::PrepareFinalWorkSize(WorkSize gws, WorkSize lws)
{
    for (uint8_t d = 0; d < gws.dimensions; d++) {
        uint16_t rem = gws.values[d] % lws.values[d];
        if (rem != 0) {
            gws.values[d] += lws.values[d] - rem;
        }
    }

    return gws;
}

}       // namespace ocl
