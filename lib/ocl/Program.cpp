#include "Program.h"

namespace ocl {

Program::Program(cl_command_queue command_queue, cl_program program, cl_device_id device) : program_(program), device_(device), command_queue_(command_queue) {
    CL_Check(clRetainCommandQueue(command_queue_));
}

Program::~Program()
{
    if (program_) {
        clReleaseProgram(program_);
    }
    if (command_queue_) {
        clReleaseCommandQueue(command_queue_);
    }
}

std::unique_ptr<Kernel> Program::CreateKernel(std::string name) {
    cl_int clError;
    cl_kernel kernel = clCreateKernel(program_, name.c_str(), &clError);
    CL_ENSURE_SUCCESS(clError, "Failed to create kernel '" << name << "'", nullptr);

    return std::unique_ptr<Kernel>(new Kernel(command_queue_, kernel, device_));
}

}       // namespace ocl
