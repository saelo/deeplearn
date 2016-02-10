# TODO

tl;dr a lot ;)

## General

* Implement support for HDF5
* Add better error handling, e.g. by defining some status enum

## Performance

* Enable out-of-order execution on the command queue
    * Maybe add helper classes InputBuffer and OutputBuffer to add events to output buffers and use events from input buffers in Kernel::Run
* Process whole mini-batches at once, i.e. make the layers accept a tensor of shape (num_batches, x, y, z, ...)
* Further optimize the kernels used by dense and convolution layers

## Computational Improvements

* Deal with numerical instabilities
    * E.g. exp(70.f) on the GPU already yields inf
* Improve weight initialization

## New Features

* Add support for optimizers other than stochastic gradient descent
