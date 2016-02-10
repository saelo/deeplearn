/*
 * OpenCL Program.
 */

#ifndef __PROGRAM_H__
#define __PROGRAM_H__

#include <memory>

#include "Utils.h"
#include "Kernel.h"

namespace ocl {

class Program {
  public:
    Program(cl_command_queue command_queue, cl_program program, cl_device_id device);

    ~Program();

    //
    // Creates a kernel object for the kernel with the specified name in this program.
    //
    std::unique_ptr<Kernel> CreateKernel(std::string name);

  private:
    // Handle to the underlying OpenCL program.
    cl_program program_;

    // Handle to the device that this program was compiled for.
    cl_device_id device_;

    // Handle to the OpenCL command queue to communicate with the device.
    // Will be retained (to increase its refcount) upon construction and released upon destruction.
    cl_command_queue command_queue_;


    DISALLOW_COPY_AND_ASSIGN(Program);
};

}   // namespace ocl

#endif
