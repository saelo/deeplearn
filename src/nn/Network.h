//
// Neural Network
//
// Copyright (c) 2016 Samuel Groß
//

#ifndef __NETWORK_H__
#define __NETWORK_H__

#include <vector>

#include "nn/Tensor.h"
#include "nn/Layer.h"
#include "nn/Activation.h"
#include "nn/Objective.h"
#include "common/Common.h"

namespace nn {

template <typename Tensor>
class Network {
    typedef Objective<Tensor> Objective;
    typedef Layer<Tensor> Layer;
    typedef Activation<Tensor> Activation;

  public:
    Network(Objective* objective) : objective_(objective), final_activation_(nullptr) { }

    ~Network()
    {
        for (Layer* layer : layers_) {
            delete layer;
        }

        if (objective_)
            delete objective_;
    }

    // Train the network on the supplied data.
    void Train(Tensor& data, Tensor& labels, Tensor& test_data, Tensor& test_labels, size_t num_epochs, size_t batch_size, float epsilon)
    {
        Assert(data.shape(0) == labels.shape(0));
        Assert(test_data.shape(0) == test_labels.shape(0));

        //
        // TODOs
        // * Add a timer here
        // * Better console output
        //

        size_t n = data.shape(0);

        for (size_t epoch = 0; epoch < num_epochs; epoch++) {
            loss_ = 0, hits_ = 0, current_iteration_ = 0;

            // We might miss a couple of inputs at the end, but that's ok since the input is shuffled.
            for (size_t batch = 0; batch < n / batch_size; batch++) {
                ProcessMiniBatch(data, labels, batch_size, epsilon);

                double loss_avg = loss_ / current_iteration_;
                double acc_avg  = hits_ / current_iteration_;
                printf("%zu/%zu  loss: %.2f  acc: %.2f\n", current_iteration_, n, loss_avg, acc_avg);
            }

            // Epoch done, evaluate performance on test data.
            double correct_count = 0;
            for (size_t test = 0; test < test_data.shape(0); test++) {
                const Tensor& output = Evaluate(test_data[test]);
                if (argmax(output) == argmax(test_labels[test]))
                    correct_count++;
            }

            std::cout << "----------------------------------------------------------------------------------------------------" << std::endl;
            std::cout << "EPOCH " << epoch + 1 << " FINISHED. ACCURACY: " << correct_count << "/" << test_data.shape(0) << " (" << correct_count / test_data.shape(0) << ")" << std::endl;
            std::cout << "----------------------------------------------------------------------------------------------------" << std::endl;
        }
    }

    // Evaluate the network's output for the given input.
    const Tensor& Evaluate(const Tensor& input)
    {
        Assert(layers_.size() > 0);
        Assert(input.shape() == layers_[0]->InputTensorShape());

        const Tensor* current = &input;

        for (Layer* layer : layers_) {
            current = &layer->Forward(*current);
        }

        return *current;
    }

    // Returns the number of layers in this network.
    size_t num_layers() const
    {
        return layers_.size();
    }

    // Returns the current input tensor shape of this network.
    //
    // This is the input tensor shape of the first layer.
    Shape InputTensorShape() const
    {
        Assert(layers_.size() > 0);
        return layers_.front()->InputTensorShape();
    }

    // Returns the current output tensor shape of this network.
    //
    // This is the output tensor shape of the last layer.
    Shape OutputTensorShape() const
    {
        Assert(layers_.size() > 0);
        return layers_.back()->OutputTensorShape();
    }

    // Appends the given layer to the end of this network.
    //
    // The caller needs to verify that the expected input shape of the
    // provided layer matches the current output shape of the network.
    // The network takes ownership of all its layers.
    void Append(Layer* layer)
    {
        Check(layers_.empty() || layer->InputTensorShape() == OutputTensorShape(), "Layer not compatible: Input tensor shape doesn't match current output tensor shape");
        layers_.push_back(layer);
    }

    // Appends an activation to the end of this network.
    void Append(Activation* activation)
    {
        Check(layers_.empty() || activation->InputTensorShape() == OutputTensorShape(), "Activation not compatible: Input tensor shape doesn't match current output tensor shape");
        final_activation_ = activation;
        layers_.push_back(activation);
    }

    // Convenience operator to append layers to a network.
    Network& operator<<(Layer* layer)
    {
        Append(layer);
        return *this;
    }

    Network& operator<<(Activation* activation)
    {
        Append(activation);
        return *this;
    }

  private:
    void ProcessMiniBatch(Tensor& train_data, Tensor& train_labels, size_t batch_size, float epsilon)
    {
        for (size_t i = 0; i < batch_size; i++) {
            current_iteration_++;

            size_t r = rand() % train_data.shape(0);

            Tensor& input = train_data[r];
            Tensor& label = train_labels[r];

            const Tensor& output = Evaluate(input);

            loss_ += objective_->Loss(output, label);
            hits_ += argmax(output) == argmax(label) ? 1 : 0;

            // We might be able to directly compute the gradients of the loss function wrt the
            // input of the final activation. See Objective.h for details.
            const Tensor* gradients = nullptr;
            bool skip_final_activation = false;
            if (final_activation_)
                gradients = objective_->LossGradientWrtActivationInput(final_activation_, label);
            if (!gradients)
                gradients = &objective_->LossGradientWrtNetworkOutput(output, label);
            else
                skip_final_activation = true;

            auto start_layer = skip_final_activation ? layers_.rbegin() + 1 : layers_.rbegin();
            for (auto it = start_layer; it != layers_.rend(); ++it) {
                Layer* layer = *it;
                gradients = &layer->Backward(*gradients);
            }
        }

        for (Layer* layer : layers_) {
            layer->GradientDescent(batch_size, epsilon);
        }
    }

    // Statistics for the current training epoch.
    double loss_, hits_;
    size_t current_iteration_;

    // List of all layers in this network.
    // The pointers are owned by this instance.
    std::vector<Layer*> layers_;

    // Optimizer to use for training. Pointer is owned by this instance.
    Objective* objective_;

    // The final activation layer, if any.
    Activation* final_activation_;

    DISALLOW_COPY_AND_ASSIGN(Network);
};

}       // namespace nn

#endif
