#include "KernelCommon.h"

kernel void ScaledAdd(uint size, global const float* x, global const float* y, float v, global float* out)
{
    uint base = get_local_id(0) + (get_global_id(0) - get_local_id(0)) * ITEMS_PER_THREAD;

    for (uint i = 0; i < ITEMS_PER_THREAD; i++) {
        uint index = base + i * get_local_size(0);
        if (index < size) {
            out[index] = x[index] + v * y[index];
        }
    }
}

#define add(x, y) (x)+(y)
BINARY_OPERATION(Add, add);
TENSOR_SCALAR_OPERATION(ScalarAdd, add);
#undef add

#define sub(x, y) (x)-(y)
BINARY_OPERATION(Sub, sub);
TENSOR_SCALAR_OPERATION(ScalarSub, sub);
#undef sub

#define mul(x, y) (x)*(y)
BINARY_OPERATION(Mul, mul);
TENSOR_SCALAR_OPERATION(ScalarMul, mul);
#undef mul

#define div(x, y) (x)/(y)
BINARY_OPERATION(Div, div);
TENSOR_SCALAR_OPERATION(ScalarDiv, div);
#undef div

UNARY_OPERATION(Exp, exp);
UNARY_OPERATION(Log, log);
