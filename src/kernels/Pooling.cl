#include "KernelCommon.h"

kernel void MaxPool2D(uint output_width, uint output_height, uint num_channels, uint input_width, uint input_height, uint pooling_width, uint pooling_height, global const float* input, global float* output)
{
    uint2 out, in;
    out.x = get_global_id(X);
    out.y = get_global_id(Y);
    uint channel = get_global_id(Z);
    in.x = out.x * pooling_width;
    in.y = out.y * pooling_height;

    if (out.x >= output_width || out.y >= output_height || channel >= num_channels)
        return;

    float curmax = FLT_MIN;
    for (uint y = in.y; y < in.y + pooling_height; y++) {
        for (uint x = in.x; x < in.x + pooling_width; x++) {
            if (x < input_width && y < input_height)
                curmax = max(curmax, input[channel * (input_width * input_height) + y * input_width + x]);
        }
    }

    output[channel * (output_width * output_height) + out.y * output_width + out.x] = curmax;
}

kernel void MaxPool2DGradients(uint gradients_width, uint gradients_height, uint num_channels, uint image_width, uint image_height, uint pooling_width, uint pooling_height, global const float* input, global const float* gradients, global float* output)
{
    uint2 out, in;
    in.x = get_global_id(X);
    in.y = get_global_id(Y);
    uint channel = get_global_id(Z);
    out.x = in.x * pooling_width;
    out.y = in.y * pooling_height;

    if (in.x >= gradients_width || in.y >= gradients_height || channel >= num_channels)
        return;

    float curmax = FLT_MIN;
    uint max_x = out.x, max_y = out.y;
    for (uint y = out.y; y < out.y + pooling_height; y++) {
        for (uint x = out.x; x < out.x + pooling_width; x++) {
            if (x < image_width && y < image_height) {
                float v = input[channel * (image_width * image_height) + y * image_width + x];
                if (v > curmax) {
                    max_x = x, max_y = y;
                    curmax = v;
                }
            }
        }
    }

    output[channel * (image_width * image_height) + max_y * image_width + max_x] = gradients[channel * (gradients_width * gradients_height) + in.y * gradients_width + in.x];
}
