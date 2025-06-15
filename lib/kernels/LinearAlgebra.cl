#include "KernelCommon.h"

// See http://www.bealto.com/gpu-gemv_v3.html
//
// Ideally we would only enforce the first dimension of the work group size..
kernel __attribute__((reqd_work_group_size(1, 256, 1)))
kernel void MatVecMul(uint num_rows, uint num_cols, uint num_elements_per_thread, global const float* m, global const float* v, local float* cache, global float* out)
{
    uint base_col = get_global_id(COL) * num_elements_per_thread;
    uint row = get_global_id(ROW);

    // Load relevant part of the vector into local memory.
    // These should be fetched by neighboring work items, thus using the local row ID.
    for (uint i = 0; i < num_elements_per_thread; i += get_local_size(ROW)) {
        uint col = i + get_local_id(ROW);
        if (col < num_elements_per_thread) {
            if (base_col + col < num_cols)
                cache[col] = v[base_col + col];
            else
                cache[col] = 0;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    float sum = 0;

    for (uint i = 0; i < num_elements_per_thread; i++)
        if (base_col + i < num_cols && row < num_rows)
            sum += m[row * num_cols + base_col + i] * cache[i];

    if (row < num_rows)
        out[row * get_global_size(COL) + get_global_id(COL)] = sum;
}

kernel void MatVecMulReduce(uint size, uint num_entries, global const float* in, global float* out)
{
    uint x = get_global_id(0);

    if (x < size) {
        float sum = 0;
        for (uint i = 0; i < num_entries; i++)
            sum += in[x * num_entries + i];
        out[x] = sum;
    }
}

kernel __attribute__((reqd_work_group_size(256, 1, 1)))
kernel void TransposedMatVecMul(uint num_rows, uint num_cols, uint num_elements_per_thread, global const float* m, global const float* v, local float* cache, global float* out)
{
    uint base_row = get_global_id(ROW) * num_elements_per_thread;
    uint col = get_global_id(COL);

    // Load relevant part of the vector into local memory.
    // These should be fetched by neighboring work items, thus using the local col ID.
    for (uint i = 0; i < num_elements_per_thread; i += get_local_size(COL)) {
        uint row = i + get_local_id(COL);
        if (row < num_elements_per_thread) {
            if (base_row + row < num_rows)
                cache[row] = v[base_row + row];
            else
                cache[row] = 0;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    float sum = 0;

    for (uint i = 0; i < num_elements_per_thread; i++)
        if (base_row + i < num_rows && col < num_cols)
            //sum += m[col * num_cols + base_row + i] * cache[i];
            sum += m[(base_row + i) * num_cols + col] * cache[i];

    if (col < num_cols)
        out[col * get_global_size(ROW) + get_global_id(ROW)] = sum;
}

kernel void TransposedVecMul(uint num_rows, uint num_cols, global const float* v1, global const float* v2, global float* out)
{
    uint row = get_global_id(0);
    uint col = get_global_id(1);

    if (row < num_rows && col < num_cols) {
        out[row * num_cols + col] = v1[row] * v2[col];
    }
}
