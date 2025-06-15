#include <algorithm>
#include <iomanip>
#include <cmath>
#include <cfloat>
#include <sstream>
#include <string.h>

#include "nn/tensor/CpuTensor.h"
#include "nn/tensor/GpuTensor.h"

using namespace std;

namespace nn {

CPUTensor::CPUTensor() : BaseTensor({}), buffer_(nullptr) { }

CPUTensor::CPUTensor(const Shape& shape) : BaseTensor(shape)
{
    buffer_ = new float[size()];
    Check(buffer_, "Out of memory");
}

CPUTensor::~CPUTensor()
{
    if (buffer_ && !is_view())
        delete [] buffer_;
}

CPUTensor::CPUTensor(const CPUTensor& other) : BaseTensor(other)
{
#if COPYGUARD
    std::cout << "Notice: CPUTensor copy constructor called." << std::endl;
#endif
    buffer_ = new float[size()];
    Check(buffer_, "Out of memory");
    copy(other.buffer_, other.buffer_ + size(), buffer_);
}

CPUTensor& CPUTensor::operator=(const CPUTensor& other)
{
    Check(!is_view() || shape() == other.shape(), "Invalid assignment to tensor view.");

    if (this == &other)
        return *this;

    // This optimization is required so tensor views work as expected.
    if (size() != other.size()) {
        delete [] buffer_;
        buffer_ = new float[other.size()];
    }

    // Assign base class properties.
    BaseTensor::operator=(other);

    copy(other.buffer_, other.buffer_ + size(), buffer_);

    return *this;
}

string CPUTensor::ToString() const
{
    stringstream stream;

    if (rank() == 1) {
        for (size_t i = 0; i < size(); i++) {
            stream << fixed << setw(5) << setprecision(3) << buffer_[i];
            if (i != size() - 1)
                stream << ", ";
        }
        stream << endl;
    } else {
        for (size_t i = 0; i < shape(0); i++) {
            stream << SubTensor(i);
        }
        stream << endl;
    }

    return stream.str();
}

bool CPUTensor::isAlmostEqual(const CPUTensor& other, float kFloat32Epsilon) const
{
    if (shape() != other.shape())
        return false;

    for (size_t i = 0; i < size(); i++) {
        float our = buffer_[i];
        float their = other.buffer_[i];
        if (!floatEq(our, their, kFloat32Epsilon))
            return false;
    }

    return true;
}


void CPUTensor::Clear()
{
    memset(buffer_, 0, size() * sizeof(float));
}

GPUTensor CPUTensor::ToGPU() const
{
    return GPUTensor(*this);
}

CPUTensor::CPUTensor(const GPUTensor& other) : BaseTensor(other.shape())
{
    buffer_ = new float[size()];
    Check(buffer_, "Out of memory");
    other.gpu_buffer()->ReadInto(buffer_, size());
}

CPUTensor::CPUTensor(const CPUTensor& base, const Shape& new_shape) : BaseTensor(new_shape), buffer_(base.buffer_)
{
    Assert(size() == base.size());
    is_view_ = true;
}

CPUTensor::CPUTensor(const CPUTensor& base, size_t index) : BaseTensor(base.shape().ElementShape())
{
    buffer_ = base.buffer_ + index * (base.shape().ElementShape().TotalElementCount());
    is_view_ = true;
}

}       // namespace nn
