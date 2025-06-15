//
// OpenCL initialization
//
// Copyright (c) 2016 Samuel Groß
//

#ifndef __OPENCL_H__
#define __OPENCL_H__

// Chooses an available OpenCL device, initializes it and sets it as default device for the nn library.
bool InitOpenCL();

#endif
