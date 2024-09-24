//
// Base tensor class
//
// Copyright (c) 2016 Samuel Groß
//

#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <vector>
#include <cfloat>
#include <cmath>

#include "common/Common.h"
#include "nn/tensor/Shape.h"

namespace nn {

//
// Tensor classes
//
// A tensor (here) is a esentially just a generalization of a vector to multiple dimensions.
//
// There are two types of tensors: CPUTensors and GPUTensors.
//
// The CPUTensor "lives" in host memory and its data is directly accessible by the CPU.
// It has a lot of (convenience) methods, such as ToString, begin and end (iterator support)
// as well as an operator for element access.
//
// The GPUTensor on the other hand "lives" in device memory and is not directly accessible by the CPU.
// As such, the above operations would be faily expensive as they require copying data to the host first.
// Therefore, these methods are simply not available :)
// Instead, one has to explicitely copy a GPUTensor to the host by calling ToHost (or ToGPU for a CPUTensor).
//
// The tensor classes also are not compatible in any way except that they have similar methods defined.
// This is done on purpose so client code is always aware of where its tensors are currently located.
// We want to avoid implicit data copying from host to device and vice versa as much as possible.
//
// All tensors are stored row-major, i.e. the rightmost dimension is stored linearily in memory.
// This design makes it easy to create cheap views onto lower-dimensional subtensors.
//
//
// About tensor views:
// It is possible to construct a tensor that shares the same underlying memory buffer with another tensor.
// These are referred to as tensor views.
//
// There are two ways to construct such a tensor view:
//
// via NewView(const Shape&)        This returns a new const Tensor that has the same underlying memory buffer as
//                                  the original tensor but potentially a different shape.
//                                  This is mostly useful to perform cheap reshape operations on const tensors.
//
// via SubTensor(size_t) or
// via operator[]                   These are basically views onto a part of the original tensor. Their rank is always one less than
//                                  the original tensor. In contrast to tensor views, sub-tensors cannot be reshaped and canonly be
//                                  assigned to if the shape stays the same.
//
// These methods are implemented in the BaseTensor class which uses specific child class constructors to construct
// the tensor view objects.
// Some of the logic (namely the assignment operator and Reshape) currently depend on the first kind of tensor views (NewView(const Shape&))
// to always be const. If that changes than these methods need to be modified as well.
//


// Constants for shape indices.
// Tensors are stored in row-major order, so cartesian coordinates are reversed.
constexpr size_t ROW = 0, COL = 1;
constexpr size_t X = 1, Y = 0;


//
// The BaseTensor class makes use of the CRTP (https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern)
// to implement some of the common tensor operators for both CPU and GPU tensors.
//
// It also manages the shape and size properties, the sub-tensor vector, as well as the flags that indicate whether this tensor is a view or sub-tensor.
//
template<class Tensor>
class BaseTensor {
  public:
    // Default constructor.
    BaseTensor(const Shape& shape) : is_view_(false), shape_(shape), size_(shape.TotalElementCount()) { }

    // Copy constructor and assignment operator.
    //
    // Assigning to a subtensor only works if the shape stays the same, in which case the part
    // of the original tensor is modified as well (so that `matrix[i] = row;` works as expected).
    //
    // There are currently no move semantic versions of these.
    // While possible to implement them, care must be taken to correctly
    // handle the case where either |this| or |other| is a tensor view.
    BaseTensor(const BaseTensor& other) : is_view_(false), shape_(other.shape_), size_(other.size_) { }
    BaseTensor& operator=(const BaseTensor& other)
    {
        shape_ = other.shape_;
        size_ = other.size_;
        return *this;
    }

    // Frees all tensor views associated with this tensor.
    virtual ~BaseTensor()
    {
        for (auto view : views_)
            if (view)
                delete view;
    }

    // Reshapes this tensor.
    //
    // Reshaping is only possible if the total number of elements stays the same.
    // This operations is cheap, i.e. no data is moved or copied.
    //
    // We disallow reshaping of views so that `tensor[i].Reshape(..)` fails. That's
    // what `NewView(const Shape&)` is for.
    bool Reshape(const Shape& new_shape)
    {
        FAIL_IF(new_shape.TotalElementCount() != shape_.TotalElementCount(), "New shape must have same total number of elements.", false);
        FAIL_IF(is_view(), "Cannot reshape tensor views.", false);

        shape_ = new_shape;
        return true;
    }

    // Returns the shape of this tensor.
    const Shape& shape() const { return shape_; }

    // Shortcut for shape()[i];
    size_t shape(size_t i) const { return shape_[i]; }

    // Returns the total number of elements stored in this tensor.
    size_t size() const { return size_; }

    // Returns the rank of this tensor.
    //
    // The rank of a tensor is the number of indices required to address
    // every element of the tensor.
    size_t rank() const { return shape_.rank(); }

    // Is this a tensor view onto another tensor?
    bool is_view() const { return is_view_; }

    // Create a view onto this tensor with a different shape.
    const Tensor* NewView(const Shape& new_shape) const
    {
        Assert(new_shape.TotalElementCount() == size());
        return new Tensor(*static_cast<const Tensor*>(this), new_shape);
    }

    // Obtain views onto a part of the original tensor.
    Tensor& SubTensor(size_t i)
    {
        Assert(rank() > 1);
        Assert(i < shape(0));

        if (views_.size() <= i)
            views_.resize(i + 1);

        if (!views_[i])
            // Not yet initialized.
            // Construct the tensor with the subtensor constructor.
            views_[i] = new Tensor(*static_cast<Tensor*>(this), i);

        return *views_[i];
    }

    const Tensor& SubTensor(size_t i) const
    {
        return const_cast<BaseTensor*>(this)->SubTensor(i);
    }

    Tensor& operator[](size_t i)
    {
        return SubTensor(i);
    }

    const Tensor& operator[](size_t i) const
    {
        return SubTensor(i);
    }

    //
    // Arithmetik operations. These are performed elementwise.
    //
    // Unfortunately we need some static_casts here to downcast 'this'
    // to the correct child class. We can safely do this as the Tensor
    // template type is always the correct child class.
    //
    Tensor operator+(const Tensor& other) const
    {
        Tensor result(shape_);
        add(*static_cast<const Tensor*>(this), other, result);
        return result;
    }

    Tensor operator-(const Tensor& other) const
    {
        Tensor result(shape_);
        sub(*static_cast<const Tensor*>(this), other, result);
        return result;
    }

    Tensor operator*(const Tensor& other) const
    {
        Tensor result(shape_);
        mul(*static_cast<const Tensor*>(this), other, result);
        return result;
    }

    Tensor operator/(const Tensor& other) const
    {
        Tensor result(shape_);
        div(*static_cast<const Tensor*>(this), other, result);
        return result;
    }

    Tensor operator+(float v) const
    {
        Tensor result(shape_);
        add(*static_cast<const Tensor*>(this), v, result);
        return result;
    }

    Tensor operator-(float v) const
    {
        Tensor result(shape_);
        sub(*static_cast<const Tensor*>(this), v, result);
        return result;
    }

    Tensor operator*(float v) const
    {
        Tensor result(shape_);
        mul(*static_cast<const Tensor*>(this), v, result);
        return result;
    }

    Tensor operator/(float v) const
    {
        Tensor result(shape_);
        div(*static_cast<const Tensor*>(this), v, result);
        return result;
    }

    Tensor& operator+=(const Tensor& other)
    {
        return add(*static_cast<Tensor*>(this), other, *static_cast<Tensor*>(this));
    }

    Tensor& operator-=(const Tensor& other)
    {
        return sub(*static_cast<Tensor*>(this), other, *static_cast<Tensor*>(this));
    }

    Tensor& operator*=(const Tensor& other)
    {
        return mul(*static_cast<Tensor*>(this), other, *static_cast<Tensor*>(this));
    }

    Tensor& operator/=(const Tensor& other)
    {
        return div(*static_cast<Tensor*>(this), other, *static_cast<Tensor*>(this));
    }

    Tensor& operator+=(float v)
    {
        return add(*static_cast<Tensor*>(this), v, *static_cast<Tensor*>(this));
    }

    Tensor& operator-=(float v)
    {
        return sub(*static_cast<Tensor*>(this), v, *static_cast<Tensor*>(this));
    }

    Tensor& operator*=(float v)
    {
        return mul(*static_cast<Tensor*>(this), v, *static_cast<Tensor*>(this));
    }

    Tensor& operator/=(float v)
    {
        return div(*static_cast<Tensor*>(this), v, *static_cast<Tensor*>(this));
    }

  protected:
    // This is protected so that child constructors can set it if needed.
    bool is_view_;

  private:
    Shape shape_;

    // Total number of elements in this tensor.
    size_t size_;

    // Views onto the elements of this tensor.
    std::vector<Tensor*> views_;
};


// GPU is using 32 bit floats, CPU is internally using 80 bit floats...
constexpr float kFloatMaxAbsDiff = 0.001;
constexpr float kFloatMaxRelDiff = 0.01;

// Float comparison to use when comparing elements of a tensor.
// Allows a certain percentual (of the maxium of f1 and f2) difference.
inline bool floatEq(float f1, float f2)
{
    if (f1 == f2)
        return true;

    float max = std::fmax(std::fabs(f1), std::fabs(f2));
    float diff = std::fabs(f1 - f2);

    if (max < 0.01) {
        // Relative error doesn't work well near zero. Use absolute error.
        return diff <= kFloatMaxAbsDiff;
    } else {
        // Relative error for larger floats.
        return diff <= max * kFloatMaxRelDiff;
    }
}


}       // namespace nn

#endif
