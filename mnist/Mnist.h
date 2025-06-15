//
// MNIST helper code
//
// Copyright (c) 2016 Samuel Groß
//

#ifndef __MNIST_H__
#define __MNIST_H__

#include "nn/Tensor.h"

namespace utils {

// Load the MNIST training set from the provided directory.
//
// Labels are converted to "one-hot" vectors of size 10, with all entries 0 except for the correct entry which will be set to 1.
// Image data will be loaded into a 28x28 tensor, with pixels converted to a float value in the range [0, 1].
//
// The results will be tensors of shape [N, 28, 28], where N is 60000 for training data and 10000 for test data.
bool LoadMNIST(std::string mnist_dir, nn::CPUTensor* train_data, nn::CPUTensor* train_labels, nn::CPUTensor* test_data, nn::CPUTensor* test_labels);

}       // namespace mnist

#endif
