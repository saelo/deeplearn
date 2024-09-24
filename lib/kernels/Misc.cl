kernel void MSE(uint size, global const float* a1, global const float* a2, global float* out)
{
    uint id = get_global_id(0);
    if (id < size) {
        out[id] = pown((a1[id] - a2[id]), 2);
    }
}
