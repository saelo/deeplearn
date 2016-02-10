/*
 * OpenCL Kernel.
 */

#ifndef __KERNEL_H__
#define __KERNEL_H__

#include <memory>

#include "Utils.h"
#include "Buffer.h"

namespace ocl {

// Helper structure to represent a local memory buffer argument for an OpenCL kernel.
// This needs to be a custom datatype so BindNextArgument() can be specialized for it.
struct LocalMemory {
    explicit LocalMemory(size_t size) : size(size) { }

    size_t size;
};

class Kernel {
  public:
    Kernel(cl_command_queue command_queue, cl_kernel kernel, cl_device_id device);

    ~Kernel();

    struct WorkSize {
        WorkSize(size_t x) : dimensions(1) {
            values[0] = x;
        }
        WorkSize(size_t x, size_t y) : dimensions(2) {
            values[0] = x;
            values[1] = y;
        }
        WorkSize(size_t x, size_t y, size_t z) : dimensions(3) {
            values[0] = x;
            values[1] = y;
            values[2] = z;
        }

        unsigned int dimensions;
        size_t values[3];
    };

    // Returns the preferred work group size multiple for this kernel.
    size_t PreferredWorkSizeMultiple() const;

    // Bind the next kernel argument.
    //
    // This is supported for all primitive data types, as well as Buffer pointers and LocalMemory instances.
    template <typename T>
    bool BindNextArgument(T value)
    {
        cl_int clErr = clSetKernelArg(kernel_, cur_index_, sizeof(T), (void *)&value);
        CL_ENSURE_SUCCESS(clErr, "Failed to bind argument for kernel", false);
        cur_index_++;
        return true;
    }

    // Calculate a decent local work size for the provided global work size and the current device.
    //
    // This method is also used when calling Run() without a local work size.
    static WorkSize CalculateLocalWorkSize(WorkSize gws);

    // Execute this kernel with the provided work size.
    //
    // This method takes care of extending the global work size to a multiple
    // of the local work size. The caller must ensure that the kernel code
    // handles out-of-bounds global IDs correctly.
    bool Run(WorkSize gws, WorkSize lws);

    // Execute this kernel and choose a fitting local work size.
    //
    // This method takes care of choosing a decent local work size for the
    // device it is running on. As above, this can lead to out-of-bounds
    // global IDs in the kernel.
    bool Run(WorkSize gws);

    // Convinience function that make use of templating to automatically bind
    // the provided arguments and run the kernel.
    // With this executing an OpenCL kernel "feels like" using a native C++ function.
    //
    //      kernel->Run(WorkSize(100), buffer, LocalMemory(20), 1337, 4.2f);
    //
    template <typename Head, typename ...Tail>
    bool Run(WorkSize gws, WorkSize lws, Head head, Tail... tail)
    {
        FAIL_IF(!BindNextArgument(head), "Could not bind argument", false);
        return Run(gws, lws, tail...);
    }

    template <typename Head, typename ...Tail>
    bool Run(WorkSize gws, Head head, Tail... tail)
    {
        FAIL_IF(!BindNextArgument(head), "Could not bind argument", false);
        return Run(gws, tail...);
    }

  private:
    // Ensure that the global work size is XXX
    WorkSize PrepareFinalWorkSize(WorkSize gws, WorkSize lws);

    // Handle to the underlying OpenCL kernel.
    cl_kernel kernel_;

    // Handle to the device that this kernel will execute on.
    cl_device_id device_;

    // Index of the next argument to be bound.
    size_t cur_index_;

    // Handle to the OpenCL command queue to communicate with the device.
    // Will be retained (to increase its refcount) upon construction and released upon destruction.
    cl_command_queue command_queue_;

    DISALLOW_COPY_AND_ASSIGN(Kernel);
};

}   // namespace ocl

#endif
