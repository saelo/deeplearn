#include "Initializer.h"

using namespace std;

namespace nn {

float ZeroInitializer()
{
    return 0.f;
}

float OneInitializer()
{
    return 1.f;
}


RandomInitializer::RandomInitializer(float mean, float stddev) : generator_(rand()), distribution_(mean, stddev) { }

float RandomInitializer::operator()()
{
    return distribution_(generator_);
}


// Specialized for ReLU activations: 1/2 of the neurons don't fire on average.
// TODO this isn't "real" Glorot initialization, maybe rename..
GlorotInitializer::GlorotInitializer(float n) : generator_(rand()), distribution_(0.f, 2 / n) { }

float GlorotInitializer::operator()()
{
    return distribution_(generator_);
}

}       // namespace nn
