//
// Tensor class
//
// Copyright (c) 2016 Samuel Groß
//

#ifndef __GPU_TENSOR_H__
#define __GPU_TENSOR_H__

#include <vector>
#include <memory>
#include <cstddef>

#include "nn/tensor/BaseTensor.h"
#include "nn/Initializer.h"
#include "nn/Gpu.h"
#include "common/Common.h"
#include "ocl/Device.h"

namespace nn {

class CPUTensor;

// A tensor located on the GPU.
class GPUTensor : public BaseTensor<GPUTensor> {
  public:
    // Creates an empty tensor. Useful to declare local variables, then assign
    // "real" values to them later on.
    GPUTensor();

    // Allocates a memory buffer but does not initialize its content.
    explicit GPUTensor(const Shape& shape);

    // Initialization constructor.
    // Initializes all values using the provided initializer.
    template <class Initializer>
    GPUTensor(const Shape& shape, Initializer initializer) : BaseTensor(shape)
    {
        float* buf = new float[size()];
        for (size_t i = 0; i < size(); i++)
            buf[i] = initializer();

        buffer_ = GPUContext::device->AllocateBuffer(size() * sizeof(float)).release();
        Check(buffer_, "Out of device memory");
        buffer_->Write(buf);
        delete [] buf;
    }

    // Destructor.
    virtual ~GPUTensor();

    // Copy constructor and assignment operator. See the comments in BaseTensor.h about these.
    GPUTensor(const GPUTensor& other);
    GPUTensor& operator=(const GPUTensor& other);

    // Returns the associated GPU buffer.
    // We could make this private and "whitelist" all tensor operations, but doesn't seem worth it.
    ocl::Buffer* gpu_buffer() const { return buffer_; }

    // Sets all elements to zero.
    void Clear();

    // Transfer the data of this tensor to a new tensor located on the host.
    CPUTensor ToHost() const;

  private:
    // Tensor view constructors.
    GPUTensor(const GPUTensor& base, const Shape& new_shape);
    GPUTensor(const GPUTensor& base, size_t index);

    // Transfer constructor.
    // Creates a GPU tensor with the data and shape of the provided CPU tensor.
    explicit GPUTensor(const CPUTensor& tensor);

    // Underlying (GPU) buffer. Pointer is owned by this instance if !dependent_.
    ocl::Buffer* buffer_;

    friend class CPUTensor;
    friend class BaseTensor;
};

// Make CPUTensors easily printable to various streams.
std::ostream& operator<<(std::ostream& os, const GPUTensor& tensor);

// Define all GPUTensor operations
#define Tensor GPUTensor
#include "nn/tensor/TensorOps.h"
#undef Tensor

}       // namespace nn

#endif
