# Deep Learning

## What?

A simple, lightweight, fully OpenCL capable toolkit for training artificial neural networks and doing some general tensor arithmetic.

## Why?

A university assignment.

I wanted to understand the details and underlying math of how neural networks work internally, so I choose this as my topic for a GPU computing freestyle assignment.

## So?

In general, if you "just" want to train some neural networks then this code isn't for you. Go ahead and use [TensorFlow](https://www.tensorflow.org), [Keras](http://keras.io), [CNTK](https://github.com/Microsoft/CNTK) or one of the other frameworks out there :)

On the other hand, if you

* Know some C++ and want to hack on some fairly simple deeplearning toolkit, or
* Want to understand the math behind neural networks, or
* Want something to experiment with that has full OpenCL support

then this code might still be interesting for you :)

## Examples

General tensor usage

```C++
// Create a tensor in host memory of shape (10, 20).
CPUTensor m({10, 20}, RandomInitializer());
// Create some 1D tensors/vectors.
CPUTensor v1({10}), v2({20}, RandomInitialzer()), v3({10});

// Fill v3 with some "arbitrary" data.
for (float& f : v3)
    f = 42;

// Matrix-vector multiplication: v1 = m * v2.
matvecmul(m, v2, v1);

// Add v3 to v1.
v1 += v3;

// Print a string representation of v1.
cout << v1 << endl;
```

It is also possible to access the lower-dimensional sub-objects of a tensor

```C++
// Create a 2D tensor/matrix.
CPUTensor m({4, 20}, ZeroInitializer);

// Some random operations on the rows of the matrix.
m[0] = CPUTensor({20}, OneInitializer);
m[1] += m[0] + CPUTensor({20}, RandomInitializer());
m[2] = m[1] * m[1];             // Elementwise multiplication
m[3] = log(m[2]);               // Elementwise natural logarithm
```

Performing a 2D convolution on the GPU

```C++
// Load an image and convolution kernel from somewhere.
CPUTensor host_image = LoadSomeImage();
CPUTensor host_conv_kernel = LoadSomeConvKernel();

// Move both tensors to the GPU.
GPUTensor image = host_image.ToGPU(), conv_kernel = host_conv_kernel.ToGPU();

// Allocate an output tensor on the GPU.
GPUTensor output(image.shape());

// Perform a 2D convolution on the GPU.
convolution(image, conv_kernel, output);

// Print out the result.
cout << output << endl;
```

See `nn/tensor/BaseTensor.h` and `nn/tensor/TensorOps.h` for an overview of the supported
tensor operations.

---

Now some actual learning: load the MNIST dataset and train a neural network on that data

```C++
// We'll train on the GPU.
using namespace nn::gpu;

CPUTensor train_data, train_labels, testing_data, test_labels;
LoadMnist(&train_data, &train_labels, &testing_data, &test_labels);

// Move the data to the GPU
GPUTensor train_data_gpu = train_data.ToGPU(),
          train_labels_gpu = train_labels.ToGPU(),
          test_data_gpu = test_data.ToGPU(),
          test_labels_gpu = test_labels.ToGPU();

Network network(new MSE(10));
network << new ReshapeLayer({28, 28}, {784})
        << new DenseLayer(784, 512)
        << new BiasLayer({512})
        << new SigmoidActivation({512})
        << new DenseLayer(512, 10)
        << new BiasLayer({10})
        << new SigmoidActivation({10});

network.Train(train_data_gpu, train_labels_gpu, test_data_gpu, test_labels_gpu, 10, 16, 0.1f);
// This will get us to around 91% accuracy. We can do better than that.
```

Same as above, but this time use a convolutional network

```C++
// ... Same as above ...

Network network(new CrossEntropy({10}));
network << new ReshapeLayer({28, 28}, {1, 28, 28})

        << new ConvolutionLayer({1, 28, 28}, 32, 3, 3)
        << new ReLUActivation({32, 28, 28})
        << new MaxPool2DLayer({32, 28, 28}, 2, 2)

        << new ConvolutionLayer({32, 14, 14}, 32, 3, 3)
        << new ReLUActivation({32, 14, 14})
        << new MaxPool2DLayer({32, 14, 14}, 2, 2)

        << new ReshapeLayer({32, 7, 7}, {32*7*7})
        << new DenseLayer(32*7*7, 1024)
        << new BiasLayer({1024})
        << new ReLUActivation({1024})

        << new DenseLayer(1024, 10)
        << new SoftmaxActivation({10});

network.Train(train_data_gpu, train_labels_gpu, test_data_gpu, test_labels_gpu, 10, 16, 0.001f);
// This will already get us to around 99% accuracy. Not too bad :)
```

## Usage

```bash
# Fetch the MNIST dataset ..
./fetch_mnist.sh

# .. build everything ..
mkdir build && cd build
cmake ../
make

# .. and train!
./deeplearn
```

## How to Neural Network

A neural network consists of a set of layers. Each layer performs some kind of computation on its inputs and (if any) its learnable weights and passes the result on to the next layer. That's the *forward pass*.

For classification tasks the output of the final layer will usually be a kind of probability distribution, representing the networks current estimate of the class of the input. For example, if we want to classify handwritten digits, then the output would be a vector with 10 elements, one for each digit. The output [0, 0.3, 0, 0, 0, 0, 0, 0.7, 0, 0] could then be interpreted as "The network is 70% sure that the input is a '7', but it could (with 30% probability) also be a '1'".

Since we know the correct answer for each input during training, we can calculate a *loss function* of the output. This could simply be the mean-squared-error or the [cross-entropy](https://en.wikipedia.org/wiki/Cross_entropy). The loss function will tell us how "wrong" the network was.

Now, the interesting part. For training the network we want to know for every trainable weight in the network how much we could improve the loss function (ultimately we want the loss function to become 0) by changing that weight. We thus need the gradient of the loss function with regard to every weight in our network.

While we could calculate the gradient independently for every weight, there's a better way: the *backpropagation algorithm*. The main idea behind the backpropagation algorithm is to calculate the gradients layer by layer, starting at the output layer. This is possible due to the chain rule: Let's say we want to calculate the gradient of f(g(x)) (you can think of f and g as being two layers in a network). The chain rule states that (f(g(x)))' = f'(g(x)) * g'(x), which means that if we get f'(g(x)) from the next layer, we simply need to calculate g(x) (which is easy, we still have x from the forward pass) and multiply them. The result then becomes the input to the next layer.

With that the required steps that each layer needs to do during the *backward pass*:

1. It will receive the gradients of the loss function wrt to its output
2. It needs to compute the gradients of the loss function wrt to all of its weights
3. It needs to compute the gradients of the loss function wrt to its input and return that. This will be the input to the previous layer.

Basically the chain rule allows us to deal with every layer/operation in our network in isolation when calculating the gradients.

Once we have all the gradients we will do what is called *gradient descent*. We will move each weight slightly in the opposite direction of its gradient, thus "moving down" on the multidimensional plane given by the loss function.
Afterwards our network will hopefully perform a little bit better.

And now we repeat this.

Further reading:
* http://neuralnetworksanddeeplearning.com
* http://www.deeplearningbook.org
