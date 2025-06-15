//
// Abstraction around an opaque GPU buffer.
//
// Copyright (c) 2016 Samuel Groß
//

#ifndef __GPU_BUFFFER_H__
#define __GPU_BUFFFER_H__

#include <memory>

#include "Utils.h"

namespace ocl {

class BufferView;

// Abstract class to represent an OpenCL buffer.
class Buffer {
  public:
    Buffer(size_t size) : size_(size) { }

    virtual ~Buffer() { }

    // Read a certain amount of bytes from this buffer into a host buffer.
    virtual bool Read(uint8_t* buffer, size_t nbytes, std::size_t offset, bool blocking) = 0;

    // Writes the content of the host buffer into the device buffer at the given offset.
    virtual bool Write(uint8_t* buffer, size_t nbytes, std::size_t offset, bool blocking) = 0;

    // Clears all bytes in the range [offset, offset + length).
    virtual void Clear(size_t offset, size_t length) = 0;

    // Creates a new view onto this buffer. A view shares the same underlying
    // memory with the original buffer.
    virtual std::unique_ptr<Buffer> NewView(size_t offset, size_t size) = 0;

    // This essentially creates an alias for the original buffer.
    std::unique_ptr<Buffer> NewView() { return NewView(0, size()); }

    // Clears the whole buffer.
    void Clear() { Clear(0, size_); }

    // Returns the size of this buffer in bytes.
    size_t size() const { return size_; }

    // Reads the specified number of elements from this device buffer into a newly allocated host buffer.
    //
    // Returns nullptr upon failure.
    template <typename T>
    T* Read(size_t nelems = 0, std::size_t offset = 0)
    {
        if (nelems == 0) {
            Assert(size_ % sizeof(T) == 0);
            nelems = size_ / sizeof(T);
        }
        T* buffer = new T[nelems];
        FAIL_IF(!buffer, "Failed to allocate memory", nullptr);

        if (!Read((uint8_t*)buffer, nelems * sizeof(T), offset, true)) {
            delete [] buffer;
            return nullptr;
        }
        return buffer;
    }

    // Reads the buffer content into the provided buffer.
    //
    // If |nelems| is zero (the default), the whole buffer content will be read.
    // If |offset| is zero (the default), reading will start at the beginning of the buffer.
    template <typename T>
    bool ReadInto(T* buffer, size_t nelems = 0, std::size_t offset = 0)
    {
        if (nelems == 0) {
            Assert(size_ % sizeof(T) == 0);
            nelems = size_ / sizeof(T);
        }
        return Read((uint8_t*)buffer, nelems * sizeof(T), offset, true);
    }

    // Writes the specified number of elements from the host buffer into this GPU buffer.
    template <typename T>
    bool Write(T* buffer, size_t nelems = 0, std::size_t offset = 0)
    {
        if (nelems == 0) {
            Assert(size_ % sizeof(T) == 0);
            nelems = size_ / sizeof(T);
        }
        Assert((offset + nelems) * sizeof(T) <= size_);
        return Write((uint8_t*)buffer, nelems * sizeof(T), offset, true);
    }

  private:
    // Returns the OpenCL handle to the underlying buffer.
    virtual cl_mem cl_buffer() const = 0;

    // Size of this buffer in bytes.
    size_t size_;

    // class Kernel is a friend class so it can access the OpenCL buffer handle.
    friend class Kernel;
    // class BufferView is a friend class so it can call cl_buffer() on the underlying buffer.
    friend class BufferView;
};


// Standard Buffer implementation based on OpenCL buffers.
class CLBuffer : public Buffer {
  public:
    CLBuffer(cl_command_queue command_queue, cl_mem buffer, size_t size);

    virtual ~CLBuffer();

    virtual bool Read(uint8_t* buffer, size_t nbytes, std::size_t offset, bool blocking) override;

    virtual bool Write(uint8_t* buffer, size_t nbytes, std::size_t offset, bool blocking) override;

    virtual void Clear(size_t offset, size_t length) override;

    virtual std::unique_ptr<Buffer> NewView(size_t offset, size_t size) override;

  protected:
    // Handle to the OpenCL command queue to communicate with the device.
    // Will be retained (to increase its refcount) upon construction and released upon destruction.
    cl_command_queue    command_queue_;

  private:
    virtual cl_mem cl_buffer() const override { return buffer_; }

    // Handle to the underlying OpenCL buffer.
    cl_mem buffer_;


    DISALLOW_COPY_AND_ASSIGN(CLBuffer);
};

// A view onto another OpenCL buffer.
//
// This class makes it easy to treat parts of a larger buffer as a standalone object.
//
class CLBufferView : public CLBuffer {
  public:
    CLBufferView(cl_command_queue command_queue, cl_mem buffer, cl_mem base_buffer, size_t size, size_t offset);

    virtual std::unique_ptr<Buffer> NewView(size_t offset, size_t size) override;

  private:
    // Handle to the original buffer.
    // We need to keep this as clCreateSubBuffer cannot take a sub-buffer as first argument.
    cl_mem base_;

    // Offset (in bytes) into the base buffer of this view.
    size_t offset_;

    DISALLOW_COPY_AND_ASSIGN(CLBufferView);
};


}       // namespace ocl

#endif
