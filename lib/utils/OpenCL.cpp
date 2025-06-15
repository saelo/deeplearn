#include <libgen.h>
#include <vector>

#include "utils/OpenCL.h"
#include "ocl/Device.h"
#include "nn/NN.h"

using namespace std;

static cl_device_id ChooseDevice()
{

    // 1. get all platform IDs
    vector<cl_platform_id> platforms;
    const cl_uint c_MaxPlatforms = 16;
    platforms.resize(c_MaxPlatforms);

    cl_uint num_platforms;
    CL_ENSURE_SUCCESS(clGetPlatformIDs(c_MaxPlatforms, &platforms[0], &num_platforms), "Failed to get CL platform ID", nullptr);
    platforms.resize(num_platforms);

    // 2. find all available GPU devices
    vector<cl_device_id> devices;
    const int max_devices = 16;
    devices.resize(max_devices);
    int total_device_count = 0;

    cl_device_type deviceType = CL_DEVICE_TYPE_GPU;

    for (size_t i = 0; i < platforms.size(); i++) {
        cl_uint num_devices_on_platform;
        clGetDeviceIDs(platforms[i], deviceType, max_devices - total_device_count, &devices[total_device_count], &num_devices_on_platform);
        total_device_count += num_devices_on_platform;
    }

    FAIL_IF(total_device_count == 0, "No device of the selected type with OpenCL support was found.", nullptr);

    // Choosing the last available device, first might be integrated graphics.
    return devices[total_device_count - 1];
}

bool InitOpenCL()
{
    cl_device_id device_id;

    device_id = ChooseDevice();
    Check(device_id, "No available OpenCL devices");
    ocl::Device* device = new ocl::Device(device_id);

    Check(device->Init(), "OpenCL device could not be initialized");

    device->PrintDeviceInfo();

    char filepath[] = __FILE__;                 // must be writable, dirname() takes a "char*", not "const char*"
    string src_directory(dirname(filepath));
    return nn::GPUContext::Init(device, src_directory + "/../kernels/");
}
