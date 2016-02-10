#ifndef __KERNEL_COMMON_H__
#define __KERNEL_COMMON_H__

// For binary and unary kernels, each thread processes this many elements of the input tensor.
#define ITEMS_PER_THREAD 10


// The following block is only processes when included from an OpenCL kernel.
#ifndef INCLUDED_BY_HOST

// We use cartesian conventions in the kernels.
#define X 0
#define Y 1
#define Z 2
#define COL 0
#define ROW 1

// Type to use for 2D positions.
// These can be negative, e.g. during halo calculations.
typedef int2 pos2;

#define UNARY_OPERATION(name, op) kernel void name(uint size, global const float* input, global float* output)      \
{                                                                                                                   \
    uint base = get_local_id(0) + (get_global_id(0) - get_local_id(0)) * ITEMS_PER_THREAD;                          \
                                                                                                                    \
    for (uint i = 0; i < ITEMS_PER_THREAD; i++) {                                                                   \
        uint index = base + i * get_local_size(0);                                                                  \
        if (index < size) {                                                                                         \
            output[index] = op(input[index]);                                                                       \
        }                                                                                                           \
    }                                                                                                               \
}

#define BINARY_OPERATION(name, op) kernel void name(uint size, global const float* x,                               \
        global const float* y, global float* output)                                                                \
{                                                                                                                   \
    uint base = get_local_id(0) + (get_global_id(0) - get_local_id(0)) * ITEMS_PER_THREAD;                          \
                                                                                                                    \
    for (uint i = 0; i < ITEMS_PER_THREAD; i++) {                                                                   \
        uint index = base + i * get_local_size(0);                                                                  \
        if (index < size) {                                                                                         \
            output[index] = op(x[index], y[index]);                                                                 \
        }                                                                                                           \
    }                                                                                                               \
}

#define TENSOR_SCALAR_OPERATION(name, op) kernel void name(uint size, global const float* x,                        \
        float v, global float* out)                                                                                 \
{                                                                                                                   \
    uint base = get_local_id(0) + (get_global_id(0) - get_local_id(0)) * ITEMS_PER_THREAD;                          \
                                                                                                                    \
    for (uint i = 0; i < ITEMS_PER_THREAD; i++) {                                                                   \
        uint index = base + i * get_local_size(0);                                                                  \
        if (index < size) {                                                                                         \
            out[index] = op(x[index], v);                                                                           \
        }                                                                                                           \
    }                                                                                                               \
}

#endif

#endif
