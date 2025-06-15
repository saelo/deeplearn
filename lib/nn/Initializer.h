//
// Tensor initializers
//
// Copyright (c) 2016 Samuel Groß
//

#ifndef __INITIALIZER_H__
#define __INITIALIZER_H__

#include <random>
#include <cstdlib>

namespace nn {

// Always returns zero.
float ZeroInitializer();

// Always returns one.
float OneInitializer();

// Returns randomly values from a distribution with the given mean and standard deviation.
class RandomInitializer {
  public:
    RandomInitializer(float mean = 0.f, float stddev = 1.f);

    // Yields the next random value.
    float operator()();
  private:
    std::mt19937 generator_;
    std::normal_distribution<float> distribution_;
};

// Weight initializer to keep the standard deviation of the input data approximately
// constant as it flows through the network.
class GlorotInitializer {
  public:
    // Construct a new GlorotInitializer.
    GlorotInitializer(float n);

    // Yields the next random value.
    float operator()();
  private:
    std::mt19937 generator_;
    std::normal_distribution<float> distribution_;
};

}       // namespace nn

#endif
