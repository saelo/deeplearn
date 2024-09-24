#include "KernelCommon.h"

// These are defined by the compiler/host code.
#if 0
#define KERNEL_WIDTH 3
#define KERNEL_HEIGHT 3
#define LOOKUP_TABLE_X { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 17, 0, 17, 0, 17, 0, 17, 0, 17, 0, 17, 0, 17, 0, 17, 0, 17, 0, 17, 0, 17, 0, 17, 0, 17, 0, 17, 0, 17, 0, 17, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 }
#define LOOKUP_TABLE_Y { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17 }
#endif

#define TILE_WIDTH 16
#define TILE_HEIGHT 16
#define KERNEL_HALFWIDTH (KERNEL_WIDTH / 2)
#define KERNEL_HALFHEIGHT (KERNEL_HEIGHT / 2)
#define KERNEL_SIZE ((KERNEL_WIDTH) * (KERNEL_HEIGHT))
#define LOOKUP_TABLE_SIZE (sizeof(halo_lookup_table_x) / sizeof(halo_lookup_table_x[0]))

#define CACHE_WIDTH (KERNEL_WIDTH * 5)
#define CACHE_HEIGHT (KERNEL_HEIGHT * 5)

// TODO get rid of this requirement by loading halo values in a loop.
#if (TILE_WIDTH + KERNEL_HALFWIDTH * 2) * (TILE_HEIGHT + KERNEL_HALFHEIGHT * 2) - (TILE_WIDTH * TILE_HEIGHT) > TILE_WIDTH * TILE_HEIGHT
  #error "Convolution kernel too large: There are more halo elements than threads!"
#endif

// Lookup tables to map a thread ID to a halo coordinate.
constant int halo_lookup_table_x[] = LOOKUP_TABLE_X;
constant int halo_lookup_table_y[] = LOOKUP_TABLE_Y;

kernel __attribute__((reqd_work_group_size(TILE_WIDTH, TILE_HEIGHT, 1)))
kernel void Convolution2D(uint width, uint height, uint channel, uint num_channels, global const float* input, global const float* conv_kernel, global float* output)
{
    // Local caches for fast memory access.
    local float tile[TILE_HEIGHT + KERNEL_HALFHEIGHT * 2][TILE_WIDTH + KERNEL_HALFWIDTH * 2];
    local float kern[KERNEL_HEIGHT][KERNEL_WIDTH];

    // Position of this thread's element in the input/output image.
    pos2 g = (pos2)(get_global_id(X), get_global_id(Y));
    uint feature_map = get_global_id(Z);

    // Local ID of this thread.
    int2 l = (int2)(get_local_id(X), get_local_id(Y));

    // Position of this thread's element in the cached tile.
    pos2 t = l + (int2)(KERNEL_HALFWIDTH, KERNEL_HALFHEIGHT);

    // Image coordinate of the upper-left element in the tile cache.
    pos2 ul = g - l - (pos2)(KERNEL_HALFWIDTH, KERNEL_HALFHEIGHT);

    // Load main area from input buffer.
    if ((uint)g.x < width && (uint)g.y < height)
        tile[t.y][t.x] = input[channel * (width * height) + g.y * width + g.x];
    else
        tile[t.y][t.x] = 0.f;

    // Load halo region from input buffer.
    uint id = get_local_id(Y) * get_local_size(X) + get_local_id(X);
    if (id < LOOKUP_TABLE_SIZE) {
        pos2 lh = (pos2)(halo_lookup_table_x[id], halo_lookup_table_y[id]);
        pos2 gh = ul + lh;
        if (gh.x >= 0 && (uint)gh.x < width && gh.y >= 0 && (uint)gh.y < height)
            tile[lh.y][lh.x] = input[channel * (width * height) + gh.y * width + gh.x];
        else
            tile[lh.y][lh.x] = 0.f;
    }

    // Load kernel into local memory and mirror it at the center.
    // TODO might want to assert that work group size > kernel size
    uint kernel_base_index = feature_map * (num_channels * KERNEL_WIDTH * KERNEL_HEIGHT) + channel * (KERNEL_WIDTH * KERNEL_HEIGHT);
    if (l.x < KERNEL_WIDTH && l.y < KERNEL_HEIGHT)
        kern[l.y][l.x] = conv_kernel[kernel_base_index + (KERNEL_HEIGHT - 1 - l.y) * KERNEL_WIDTH + (KERNEL_WIDTH - 1 - l.x)];

    // Sync threads.
    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform the convolution.
    float value = 0.f;
    for (int ky = -KERNEL_HALFHEIGHT; ky <= KERNEL_HALFHEIGHT; ky++) {
        for (int kx = -KERNEL_HALFWIDTH; kx <= KERNEL_HALFWIDTH; kx++) {
            value += tile[t.y + ky][t.x + kx] * kern[ky + KERNEL_HALFHEIGHT][kx + KERNEL_HALFWIDTH];
        }
    }

    // Write back result. This kernel is called multiple times (once per input channel)
    // and the results are added up into the final images.
    if ((uint)g.x < width && (uint)g.y < height) {
        if (channel == 0)
            output[feature_map * (width * height) + g.y * width + g.x] = value;
        else
            output[feature_map * (width * height) + g.y * width + g.x] += value;
    }
}

// Same as the convolution, except that the kernel is not mirrored.
// TODO this is pretty much excactly the same code except for the part that loads the kernel from global memory.
// For some reason the compiler doesn't like us putting the common code into a separate function though: "SC failed. No reason given."...
kernel __attribute__((reqd_work_group_size(TILE_WIDTH, TILE_HEIGHT, 1)))
kernel void CrossCorrelation2D(uint width, uint height, uint channel, uint num_feature_maps, global const float* input, global const float* conv_kernel, global float* output)
{
    // Local caches for fast memory access.
    local float tile[TILE_HEIGHT + KERNEL_HALFHEIGHT * 2][TILE_WIDTH + KERNEL_HALFWIDTH * 2];
    local float kern[KERNEL_HEIGHT][KERNEL_WIDTH];

    // Position of this thread's element in the input/output image.
    pos2 g = (pos2)(get_global_id(X), get_global_id(Y));
    uint feature_map = get_global_id(Z);

    // Local ID of this thread.
    int2 l = (int2)(get_local_id(X), get_local_id(Y));

    // Position of this thread's element in the cached tile.
    pos2 t = l + (int2)(KERNEL_HALFWIDTH, KERNEL_HALFHEIGHT);

    // Image coordinate of the upper-left element in the tile cache.
    pos2 ul = g - l - (pos2)(KERNEL_HALFWIDTH, KERNEL_HALFHEIGHT);

    // Load main area from input buffer.
    if ((uint)g.x < width && (uint)g.y < height)
        tile[t.y][t.x] = input[channel * (width * height) + g.y * width + g.x];
    else
        tile[t.y][t.x] = 0.f;

    // Load halo region from input buffer.
    uint id = get_local_id(Y) * get_local_size(X) + get_local_id(X);
    if (id < LOOKUP_TABLE_SIZE) {
        pos2 lh = (pos2)(halo_lookup_table_x[id], halo_lookup_table_y[id]);
        pos2 gh = ul + lh;
        if (gh.x >= 0 && (uint)gh.x < width && gh.y >= 0 && (uint)gh.y < height)
            tile[lh.y][lh.x] = input[channel * (width * height) + gh.y * width + gh.x];
        else
            tile[lh.y][lh.x] = 0.f;
    }

    // Load kernel into local memory and mirror it at the center.
    // Kernel has different shape here than in the Convolution2D kernel. See TensorOps.h and/or CpuTensorOps.h.
    uint kernel_base_index = channel * (num_feature_maps * KERNEL_WIDTH * KERNEL_HEIGHT) + feature_map * (KERNEL_WIDTH * KERNEL_HEIGHT);
    if (l.x < KERNEL_WIDTH && l.y < KERNEL_HEIGHT)
        kern[l.y][l.x] = conv_kernel[kernel_base_index + l.y * KERNEL_WIDTH + l.x];

    // Sync threads.
    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform the convolution.
    float value = 0.f;
    for (int ky = -KERNEL_HALFHEIGHT; ky <= KERNEL_HALFHEIGHT; ky++) {
        for (int kx = -KERNEL_HALFWIDTH; kx <= KERNEL_HALFWIDTH; kx++) {
            value += tile[t.y + ky][t.x + kx] * kern[ky + KERNEL_HALFHEIGHT][kx + KERNEL_HALFWIDTH];
        }
    }

    // Write back result. This kernel is called multiple times (once per input channel)
    // and the results are added up into the final images.
    if ((uint)g.x < width && (uint)g.y < height) {
        if (channel == 0)
            output[feature_map * (width * height) + g.y * width + g.x] = value;
        else
            output[feature_map * (width * height) + g.y * width + g.x] += value;
    }
}


#if TILE_WIDTH < KERNEL_WIDTH || TILE_HEIGHT < KERNEL_HEIGHT
  #error "Convolution kernel too large. Gradient computation will fail."
#endif

kernel __attribute__((reqd_work_group_size(KERNEL_WIDTH * KERNEL_HEIGHT, 1, 1)))
kernel void Convolution2DGradients(uint width, uint height, uint num_channels, global const float* input, global const float* gradients, global float* kernels)
{
    uint feature_map_index = get_global_id(Z);
    uint channel_index = get_global_id(Y);
    uint kernel_weight_index = get_local_id(X);

    local float local_input[TILE_HEIGHT][TILE_WIDTH];
    local float local_gradients[TILE_HEIGHT + 2 * KERNEL_HALFHEIGHT][TILE_WIDTH + 2 * KERNEL_HALFWIDTH];

    // Index delta to obtain the output element from an input element for our kernel weight.
    pos2 k = (pos2)(kernel_weight_index % KERNEL_WIDTH, kernel_weight_index / KERNEL_WIDTH);
    // Obtain the relativ delta to the center of the kernel. Also mirrors the kernel at the same time.
    k -= (pos2)(KERNEL_HALFWIDTH, KERNEL_HALFHEIGHT);


    float gradient = 0.f;

    // Iterate over all elements of the input image.
    uint x = 0, y = 0;
    while (y < height) {
        while (x < width) {
            //
            // Load next block into local memory.
            barrier(CLK_LOCAL_MEM_FENCE);
            for (uint i = kernel_weight_index; i < TILE_WIDTH * TILE_HEIGHT; i += get_local_size(X)) {
                pos2 l = (pos2)(i % TILE_WIDTH, i / TILE_WIDTH);
                pos2 g = l + (pos2)(x, y);
                if (g.x < (int)width && g.y < (int)height)
                    local_input[l.y][l.x] = input[channel_index * (width * height) + g.y * width + g.x];
                else
                    local_input[l.y][l.x] = 0.f;
            }

            for (uint i = kernel_weight_index; i < (TILE_WIDTH + 2 * KERNEL_HALFWIDTH) * (TILE_HEIGHT + 2 * KERNEL_HALFHEIGHT); i += get_local_size(X)) {
                pos2 l = (pos2)(i % (TILE_WIDTH + 2 * KERNEL_HALFWIDTH), i / (TILE_WIDTH + 2 * KERNEL_HALFWIDTH));
                pos2 g = l + (pos2)(x, y) - (pos2)(KERNEL_HALFWIDTH, KERNEL_HALFHEIGHT);
                if (g.x >= 0 && g.x < (int)width && g.y >= 0 && g.y < (int)height)
                    local_gradients[l.y][l.x] = gradients[feature_map_index * (width * height) + g.y * width + g.x];
                else
                    local_gradients[l.y][l.x] = 0.f;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            //
            // Do the "convolution".
            for (uint ly = KERNEL_HALFHEIGHT; ly < TILE_HEIGHT + KERNEL_HALFHEIGHT; ly++) {
                for (uint lx = KERNEL_HALFWIDTH; lx < TILE_WIDTH + KERNEL_HALFWIDTH; lx++) {
                    gradient += local_input[ly - KERNEL_HALFHEIGHT][lx - KERNEL_HALFWIDTH] * local_gradients[ly + k.y][lx + k.x];
                }
            }

            x += TILE_WIDTH;
        }

        x = 0;
        y += TILE_HEIGHT;
    }

    // Write back
    kernels[feature_map_index * (num_channels * KERNEL_WIDTH * KERNEL_HEIGHT) + channel_index * (KERNEL_WIDTH * KERNEL_HEIGHT) + kernel_weight_index] = gradient;
}
