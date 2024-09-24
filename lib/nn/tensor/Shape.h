//
// Shape class
//
// Copyright (c) 2016 Samuel Groß
//

#ifndef __SHAPE_H__
#define __SHAPE_H__

#include <vector>
#include <memory>
#include <iostream>

#include "common/Common.h"

namespace nn {

// Class to represent the shape of a tensor.
class Shape {
  public:
    // no 'explicit' keyword here, initializer list literals should represent shapes.
    Shape(std::initializer_list<size_t> l) : data_(l)
    {
        for (auto& d : data_) {
            Assert(d > 0);
        }
    }

    Shape(std::vector<size_t> d) : data_(d)
    {
        for (auto& d : data_) {
            Assert(d > 0);
        }
    }

    // Returns a string representation like "Shape({1, 2, 3})" for this shape.
    std::string ToString() const;

    // Calculates and returns the total number of elements that a tensor of this shape would contain.
    size_t TotalElementCount() const;

    // Returns a new shape with the first dimension removed.
    Shape ElementShape() const;

    // Returns the rank of a tensor of this shape.
    size_t rank() const { return data_.size(); }

    // Comparison operators
    bool operator==(const Shape& other) const { return data_ == other.data_; }
    bool operator!=(const Shape& other) const { return data_ != other.data_; }

    // Dimension access
    size_t operator[](size_t index) const { Assert(index < data_.size()); return data_[index]; }

  private:
    // Internally the shape is stored as a vector of size_t values, one for each dimension.
    std::vector<size_t> data_;
};

// Make Shapes easily printable to various streams.
inline std::ostream& operator<<(std::ostream& os, const Shape& shape)
{
    os << shape.ToString();
    return os;
}

}       // namespace nn

#endif
