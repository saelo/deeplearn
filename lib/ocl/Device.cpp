#include <vector>
#include <iostream>
#include <fstream>

#include "Device.h"
#include "Utils.h"

using namespace std;

namespace ocl {

Device::~Device()
{
    if (command_queue_) {
        clReleaseCommandQueue(command_queue_);
    }
    if (context_) {
        clReleaseContext(context_);
    }
    if (device_) {
        clReleaseDevice(device_);
    }
}

#define PRINT_INFO_STR(title, buffer, content_size, buffer_size, expr) { expr; buffer[content_size] = '\0'; cout << title << ": " << buffer << endl; }
#define PRINT_INFO_INT(title, num, unit, expr) { expr; cout << title << ": " << num << unit << endl; }

void Device::PrintDeviceInfo()
{
    cl_platform_id platform;
    clGetDeviceInfo(device_, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &platform, NULL);

    const int buffer_size = 1024;
    char buffer[buffer_size];
    size_t content_size;
    cl_ulong intval;

    cout << endl << "******************************" << endl << endl;
    cout << "OpenCL platform:" << endl << endl;
    PRINT_INFO_STR("Name", buffer, content_size, buffer_size, clGetPlatformInfo(platform, CL_PLATFORM_NAME, buffer_size, (void*)buffer, &content_size));
    PRINT_INFO_STR("Vendor", buffer, content_size, buffer_size, clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, buffer_size, (void*)buffer, &content_size));
    PRINT_INFO_STR("Version", buffer, content_size, buffer_size, clGetPlatformInfo(platform, CL_PLATFORM_VERSION, buffer_size, (void*)buffer, &content_size));
    PRINT_INFO_STR("Profile", buffer, content_size, buffer_size, clGetPlatformInfo(platform, CL_PLATFORM_PROFILE, buffer_size, (void*)buffer, &content_size));
    cout << endl << "Device:" << endl << endl;
    PRINT_INFO_STR("Name", buffer, content_size, buffer_size, clGetDeviceInfo(device_, CL_DEVICE_NAME, buffer_size, (void*)buffer, &content_size));
    PRINT_INFO_STR("Vendor", buffer, content_size, buffer_size, clGetDeviceInfo(device_, CL_DEVICE_VENDOR, buffer_size, (void*)buffer, &content_size));
    PRINT_INFO_STR("Driver version", buffer, content_size, buffer_size, clGetDeviceInfo(device_, CL_DRIVER_VERSION, buffer_size, (void*)buffer, &content_size));

    PRINT_INFO_INT("Global memory size", intval, " Bytes", clGetDeviceInfo(device_, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &intval, &content_size));
    PRINT_INFO_INT("Global memory cache size", intval, " Bytes", clGetDeviceInfo(device_, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(cl_ulong), &intval, &content_size));
    PRINT_INFO_INT("Local memory size", intval, " Bytes", clGetDeviceInfo(device_, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &intval, &content_size));
    PRINT_INFO_INT("Address bits", intval, "", clGetDeviceInfo(device_, CL_DEVICE_ADDRESS_BITS, sizeof(cl_ulong), &intval, &content_size));
    PRINT_INFO_INT("Compute Units", intval, "", clGetDeviceInfo(device_, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_ulong), &intval, &content_size));
    PRINT_INFO_INT("Clock Frequency", intval, " MHz", clGetDeviceInfo(device_, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_ulong), &intval, &content_size));

    cout << endl << "******************************" << endl << endl;
}

#undef PRINT_INFO_STR
#undef PRINT_INFO_INT

bool Device::Init()
{
    cl_int clError;

    context_ = clCreateContext(NULL, 1, &device_, NULL, NULL, &clError);
    CL_ENSURE_SUCCESS(clError, "Failed to create OpenCL context.", false);

    command_queue_ = clCreateCommandQueue(context_, device_, 0, &clError);
    CL_ENSURE_SUCCESS(clError, "Failed to create the command queue in the context", false);

    return true;
}

void Device::AwaitJobCompletion()
{
    CL_Check(clFinish(command_queue_));
}

size_t Device::MaxWorkGroupSize()
{
    size_t size;
    CL_Check(clGetDeviceInfo(device_, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(cl_ulong), &size, nullptr));
    return size;
}

unique_ptr<Buffer> Device::AllocateBuffer(size_t size, cl_mem_flags flags)
{
    cl_int clError;
    cl_mem buffer = clCreateBuffer(context_, flags, size, NULL, &clError);
    CL_ENSURE_SUCCESS(clError, "Failed to allocate buffer", nullptr);

    return unique_ptr<Buffer>(new CLBuffer(command_queue_, buffer, size));
}

unique_ptr<Program> Device::CreateProgram(const string& source_code, const string& compiler_args)
{
    cl_program prog = nullptr;

    const char* src = source_code.c_str();
    size_t length = source_code.size();

    cl_int clError;
    prog = clCreateProgramWithSource(context_, 1, &src, &length, &clError);
    CL_ENSURE_SUCCESS(clError, "Failed to create CL program from source.", nullptr);

    clError = clBuildProgram(prog, 1, &device_, compiler_args.c_str(), NULL, NULL);
    PrintBuildLog(prog);
    if(clError != CL_SUCCESS) {
        cerr << "Failed to build CL program." << endl;
        clReleaseProgram(prog);
        return nullptr;
    }

    return unique_ptr<Program>(new Program(command_queue_, prog, device_));
}

unique_ptr<Program> Device::CreateProgramFromFile(const string& path, const string& compiler_args)
{
    ifstream source_file;

    source_file.open(path.c_str());
    FAIL_IF(!source_file.is_open(), "Failed to open file '" << path << "'.", nullptr);

    source_file.seekg(0, ios::end);
    ifstream::pos_type file_size = source_file.tellg();
    source_file.seekg(0, ios::beg);

    string source_code;
    source_code.resize((size_t)file_size);
    source_file.read(&source_code[0], file_size);

    return CreateProgram(source_code, compiler_args);
}

void Device::PrintBuildLog(cl_program Program)
{
    cl_build_status build_status;
    clGetProgramBuildInfo(Program, device_, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &build_status, NULL);

    size_t log_size;
    clGetProgramBuildInfo(Program, device_, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

    if (log_size > 1) {
        string log(log_size+1, ' ');
        clGetProgramBuildInfo(Program, device_, CL_PROGRAM_BUILD_LOG, log_size, &log[0], NULL);
        log[log_size] = '\0';

        if (log_size > 2) {
            cout << "Build log:" << endl;
            cout << log << endl;
        }
    }

    Check(build_status == CL_SUCCESS, "Program compilation failed");
}

}       // namespace ocl
