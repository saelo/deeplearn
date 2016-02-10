#include <algorithm>
#include <iomanip>
#include <cmath>
#include <sstream>

#include "nn/tensor/GpuTensor.h"
#include "nn/tensor/CpuTensor.h"

using namespace std;

namespace nn {

GPUTensor::GPUTensor() : BaseTensor({}) { }

GPUTensor::GPUTensor(const Shape& shape) : BaseTensor(shape)
{
    buffer_ = GPUContext::device->AllocateBuffer(size() * sizeof(float)).release();
    Check(buffer_, "Out of device memory");
}

GPUTensor::GPUTensor(const GPUTensor& other) : BaseTensor(other.shape())
{
#if COPYGUARD
    std::cout << "Notice: GPUTensor copy constructor called." << std::endl;
#endif
    // TODO add CopyInto() to ocl::Buffer that uses clEnqueueCopyBuffer.
    buffer_ = GPUContext::device->AllocateBuffer(size() * sizeof(float)).release();
    float* buf = other.buffer_->Read<float>();
    buffer_->Write(buf);
    delete [] buf;
}

GPUTensor& GPUTensor::operator=(const GPUTensor& other)
{
    Check(!is_view() || shape() == other.shape(), "Invalid assignment to tensor view.");

    if (this == &other)
        return *this;

    if (shape() != other.shape()) {
        delete buffer_;
        buffer_ = GPUContext::device->AllocateBuffer(other.size() * sizeof(float)).release();
    }

    // Assign base class properties.
    BaseTensor::operator=(other);

    // TODO Same as above, implement ocl::Buffer::CopyInto().
    float* tmpbuf = other.buffer_->Read<float>();
    buffer_->Write(tmpbuf);
    delete [] tmpbuf;

    return *this;
}

GPUTensor::~GPUTensor()
{
    if (buffer_)
        delete buffer_;
}

void GPUTensor::Clear()
{
    buffer_->Clear();
}

CPUTensor GPUTensor::ToHost() const
{
    return CPUTensor(*this);
}

GPUTensor::GPUTensor(const GPUTensor& base, const Shape& new_shape) : BaseTensor(new_shape), buffer_(base.buffer_->NewView().release())
{
    Assert(size() == base.size());
    is_view_ = true;
}

GPUTensor::GPUTensor(const GPUTensor& base, size_t index) : BaseTensor(base.shape().ElementShape())
{
    size_t new_size = base.shape().ElementShape().TotalElementCount() * sizeof(float);
    buffer_ = base.buffer_->NewView(index * new_size, new_size).release();
    is_view_ = true;
}

GPUTensor::GPUTensor(const CPUTensor& tensor) : BaseTensor(tensor.shape())
{
    buffer_ = GPUContext::device->AllocateBuffer(size() * sizeof(float)).release();
    Check(buffer_, "Out of device memory");
    buffer_->Write(tensor.buffer_);
}

ostream& operator<<(ostream& os, const GPUTensor& tensor)
{
    os << tensor.ToHost().ToString();
    return os;
}

}       // namespace nn
