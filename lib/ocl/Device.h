//
// Wrapper around an OpenCL Device.
//
// Copyright (c) 2016 Samuel Groß
//

#ifndef __CL_CONTEXT_H__
#define __CL_CONTEXT_H__

#include <memory>

#include "Utils.h"
#include "Buffer.h"
#include "Program.h"

namespace ocl {

class Device {
  public:
    Device(cl_device_id device_id) : device_(device_id) { }
    ~Device();

    // Initializes this device.
    //
    // Returns false on error.
    bool Init();

    // Prints device information to stdout.
    void PrintDeviceInfo();

    // Waits for all currently queued actions for this device to be completed.
    void AwaitJobCompletion();

    // Returns a handle to the command queue for this device.
    cl_command_queue command_queue() { return command_queue_; }

    // Returns the maximum number of threads per work group for this device.
    size_t MaxWorkGroupSize();

    // Allocates a new buffer on this device.
    std::unique_ptr<Buffer> AllocateBuffer(size_t size, cl_mem_flags flags);
    std::unique_ptr<Buffer> AllocateBuffer(size_t size) { return AllocateBuffer(size, CL_MEM_READ_WRITE); }

    // Allocates a new buffer and zero initializes it.
    std::unique_ptr<Buffer> AllocateZeroFilledBuffer(size_t size) { auto buf = AllocateBuffer(size, CL_MEM_READ_WRITE); buf->Clear(); return buf; }

    // Creates a program on this device from the given source code.
    std::unique_ptr<Program> CreateProgram(const std::string& source, const std::string& compile_options);
    std::unique_ptr<Program> CreateProgram(const std::string& source) { return CreateProgram(source, ""); }

    // Loads the source code from the specified file and create a program on this device from the source code.
    std::unique_ptr<Program> CreateProgramFromFile(const std::string& path, const std::string& compile_options);
    std::unique_ptr<Program> CreateProgramFromFile(const std::string& path) { return CreateProgramFromFile(path, ""); }

  private:
    // Handle for the underlying OpenCL device.
    cl_device_id device_;

    // OpenCL context for this device. Valid after Init() has been called.
    cl_context context_;

    // OpenCL command queue for this device. Valid after Init() has been called.
    cl_command_queue command_queue_;


    void PrintBuildLog(cl_program prog);

    DISALLOW_COPY_AND_ASSIGN(Device);
};

}

#endif
