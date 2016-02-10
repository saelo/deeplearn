#include <string>
#include <sstream>
#include "nn/tensor/Shape.h"

using namespace std;

namespace nn {

string Shape::ToString() const
{
    stringstream res;

    res << "Shape({";
    string separator = "";
    for (size_t i : data_) {
        res << separator << i;
        separator = ", ";
    }
    res << "})";

    return res.str();
}

size_t Shape::TotalElementCount() const
{
    if (data_.size() == 0)
        return 0;

    size_t result = 1;

    for (size_t d : data_) {
        result *= d;
    }

    return result;
}

// Returns a new shape with the first dimension removed.
Shape Shape::ElementShape() const
{
    Assert(rank() > 1);
    return Shape(std::vector<size_t>(data_.begin() + 1, data_.end()));
}

}       // namespace nn
