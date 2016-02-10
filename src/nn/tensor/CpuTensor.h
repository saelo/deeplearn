//
// Tensor class
//
// Copyright (c) 2016 Samuel Groß
//

#ifndef __CPU_TENSOR_H__
#define __CPU_TENSOR_H__

#include <vector>
#include <type_traits>
#include <memory>
#include <cstddef>
#include <iostream>

#include "nn/tensor/BaseTensor.h"
#include "nn/Initializer.h"
#include "ocl/Device.h"
#include "common/Common.h"

namespace nn {

class GPUTensor;

// A tensor located in host memory.
class CPUTensor : public BaseTensor<CPUTensor> {
  public:
    // A simple pointer is sufficient as iterator for our purposes.
    typedef float* iterator;
    typedef const float* const_iterator;

    // Creates an empty tensor. Useful to declare local variables, then assign
    // "real" values to them later on.
    CPUTensor();

    // Default constructor for a CPUTensor. Does not initialize its data.
    explicit CPUTensor(const Shape& shape);

    // Initialization constructor.
    // Initializes all values using the provided initializer.
    template <class Initializer>
    CPUTensor(const Shape& shape, Initializer initializer) : BaseTensor(shape)
    {
        buffer_ = new float[size()];
        Check(buffer_, "Out of memory");

        for (size_t i = 0; i < size(); i++)
            buffer_[i] = initializer();
    }

    // Destructor, frees the associated buffer if this isn't a view onto another tensor.
    virtual ~CPUTensor();

    // Copy constructor and assignment operator. See the comments in BaseTensor.h about these.
    CPUTensor(const CPUTensor& other);
    CPUTensor& operator=(const CPUTensor& other);

    // Returns a string representation of this tensor.
    std::string ToString() const;

    // Element access operators.
    //
    // The number of provided indices must be equal to the rank of the tensor.
    template <typename ...Index>
    float& Element(Index... indices)
    {
        size_t linear_index = LinearIndex(0, 0, indices...);
        return buffer_[linear_index];
    }

    template <typename ...Index>
    float Element(Index... indices) const
    {
        size_t linear_index = LinearIndex(0, 0, indices...);
        return buffer_[linear_index];
    }

    // operator() is used for convenient element access.
    template <typename ...Index>
    float& operator()(Index... indices)
    {
        return Element(indices...);
    }

    template <typename ...Index>
    float operator()(Index... indices) const
    {
        return Element(indices...);
    }

    // Comparison operators.
    //
    // Two tensors are considered equal if their shape is equal and
    // floatEq(x, y) is true for each pair of elements with same index.
    bool operator==(const CPUTensor& other) const;
    bool operator!=(const CPUTensor& other) const;

    // Iterator support.
    iterator begin() { return buffer_; }
    iterator end() { return buffer_ + size(); }
    const_iterator begin() const { return buffer_; }
    const_iterator end() const { return buffer_ + size(); }

    // Sets all elements to zero.
    void Clear();

    // Transfer the data of this tensor to a new tensor located on the GPU.
    GPUTensor ToGPU() const;

  private:
    // Tensor view constructors.
    CPUTensor(const CPUTensor& base, const Shape& new_shape);
    CPUTensor(const CPUTensor& base, size_t index);

    // Linear index calculation.
    // Done with variadic templates so we get type safety as well as infinite number of arguments :)
    inline size_t LinearIndex(size_t current_index, size_t current_dimension) const
    {
        Assert(current_index < size());
        Assert(current_dimension == rank());
        return current_index;
    }
    template <typename Head, typename ...Tail>
    size_t LinearIndex(size_t current_index, size_t current_dimension, Head current, Tail... remaining) const
    {
        static_assert(std::is_integral<Head>::value, "Tensor indices must be integers.");
        Assert(current_dimension < rank());

        current_index *= shape(current_dimension);
        current_index += current;
        return LinearIndex(current_index, current_dimension + 1, remaining...);
    }

    // Transfer constructor.
    // Creates a CPU tensor with the data and shape of the provided GPU tensor.
    explicit CPUTensor(const GPUTensor& tensor);

    // Underlying (host) buffer. Pointer is owned by this instance if !is_view().
    float* buffer_;

    friend class GPUTensor;
    friend class BaseTensor;
};

// Make CPUTensors easily printable to various streams.
inline std::ostream& operator<<(std::ostream& os, const CPUTensor& tensor)
{
    os << tensor.ToString();
    return os;
}

// Define all CPUTensor operations
#define Tensor CPUTensor
#include "nn/tensor/TensorOps.h"
#undef Tensor

}       // namespace nn

#endif
