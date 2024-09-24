#include "Buffer.h"

using namespace std;

namespace ocl {

CLBuffer::CLBuffer(cl_command_queue command_queue, cl_mem buffer, size_t size) : Buffer(size), command_queue_(command_queue), buffer_(buffer)
{
    CL_Check(clRetainCommandQueue(command_queue_));
}

CLBuffer::~CLBuffer()
{
    if (buffer_) {
        clReleaseMemObject(buffer_);
    }
    if (command_queue_) {
        clReleaseCommandQueue(command_queue_);
    }
}

bool CLBuffer::Read(uint8_t* buffer, size_t nbytes, std::size_t offset, bool blocking)
{
    Assert(offset + nbytes <= size());
    CL_ENSURE_SUCCESS(clEnqueueReadBuffer(command_queue_, buffer_, blocking, offset, nbytes, buffer, 0, nullptr, nullptr), "Error reading data from device", false);
    return true;
}

bool CLBuffer::Write(uint8_t* buffer, size_t nbytes, std::size_t offset, bool blocking)
{
    Assert(offset + nbytes <= size());
    CL_ENSURE_SUCCESS(clEnqueueWriteBuffer(command_queue_, buffer_, blocking, offset, nbytes, buffer, 0, nullptr, nullptr), "Error writing data to device", false);
    return true;
}

void CLBuffer::Clear(size_t offset, size_t length)
{
    Assert(length <= size());
    // clEnqueueFillBuffer seems buggy on my AMD card .....
    //uint8_t zero = 0;
    //CL_ENSURE_SUCCESS(clEnqueueFillBuffer(command_queue_, buffer_, &zero, 1, offset, length, 0, nullptr, nullptr), "Error clearing buffer", );

    static uint8_t zeroes[1024*1024];
    size_t i = 0;
    while (length) {
        size_t to_write = std::min(length, (size_t)1024*1024);
        Write(zeroes, to_write, offset + i, false);
        length -= to_write;
        i += to_write;
    }
}

unique_ptr<Buffer> CLBuffer::NewView(size_t offset, size_t size)
{
    Assert(offset + size <= this->size());

    cl_int retval;
    cl_buffer_region region;
    region.origin = offset;
    region.size = size;

    cl_mem sub_buffer =clCreateSubBuffer(buffer_, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &retval);
    CL_ENSURE_SUCCESS(retval, "Failed to create sub-buffer", nullptr);

    return unique_ptr<Buffer>(new CLBufferView(command_queue_, sub_buffer, buffer_, size, offset));
}

CLBufferView::CLBufferView(cl_command_queue command_queue, cl_mem buffer, cl_mem base_buffer, size_t size, size_t offset) :
    CLBuffer(command_queue, buffer, size),
    base_(base_buffer),
    offset_(offset) { }

unique_ptr<Buffer> CLBufferView::NewView(size_t offset, size_t size)
{
    Assert(offset + size <= this->size());

    cl_int retval;
    cl_buffer_region region;
    region.origin = offset + offset_;
    region.size = size;
    cl_mem sub_buffer =clCreateSubBuffer(base_, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &retval);
    CL_ENSURE_SUCCESS(retval, "Failed to create sub-buffer", nullptr);

    return unique_ptr<Buffer>(new CLBufferView(command_queue_, sub_buffer, base_, size, offset));
}

}       // namespace ocl
