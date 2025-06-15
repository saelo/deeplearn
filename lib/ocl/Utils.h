/*
 * Utility functions and macros
 */

#ifndef __UTILS_H__
#define __UTILS_H__

#include <iostream>

#include "../common/Common.h"

// All OpenCL headers
#if defined(WIN32)
    #include <CL/opencl.h>
#elif defined (__APPLE__) || defined(MACOSX)
    #include <OpenCL/opencl.h>
#else
    #include <CL/cl.h>
#endif

namespace ocl {
namespace util {

	const char* GetCLErrorString(cl_int CLErrorCode);

}       // namespace util
}       // namespace ocl

#define CL_ENSURE_SUCCESS(expr, errmsg, retval) do { cl_int e=(expr); if (CL_SUCCESS!=e) { std::cerr << "Error: " << errmsg << " [" << ocl::util::GetCLErrorString(e) << "]" << std::endl; return retval; }} while(0)
#define CL_Check(expr) do { cl_int e=(expr); if (CL_SUCCESS!=e) { std::cerr << "Check failed in line " << __LINE__ << " in file " << __FILE__ ". OpenCL Error: " << ocl::util::GetCLErrorString(e) << std::endl; exit(-1); }} while(0)

#endif
