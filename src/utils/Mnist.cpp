#include "utils/Mnist.h"
#include "utils/File.h"

using namespace std;
using namespace nn;

namespace utils {

// Converts a 32 bit integer from big endian to little endian and vice versa.
static inline uint32_t swap(uint32_t v)
{
    return ((v << 24) & 0xff000000) |
           ((v << 8 ) & 0x00ff0000) |
           ((v >> 8 ) & 0x0000ff00) |
           ((v >> 24) & 0x000000ff);
}

static bool ProcessMNISTData(const string& data, CPUTensor* out)
{
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(data.c_str());
    size_t len = data.size();

    uint32_t magic = swap(*(uint32_t*)bytes);
    FAIL_IF(magic != 0x00000803, "Invalid magic for MNIST data file", false);
    bytes += 4;

    uint32_t num_images = swap(*(uint32_t*)bytes);
    FAIL_IF(num_images * 28 * 28 != len - 16, "Corrupted MNIST data file, length mismatch.", false);
    bytes += 4;

    uint32_t num_rows = swap(*(uint32_t*)bytes);
    bytes += 4;
    uint32_t num_cols = swap(*(uint32_t*)bytes);
    bytes += 4;
    FAIL_IF(num_rows != 28 || num_cols != 28, "Unsupported MNIST image dimensions", false);

    *out = CPUTensor({num_images, num_rows, num_cols});
    CPUTensor& tensor_data = *out;

    for (uint32_t i = 0; i < num_images; i++) {
        for (uint32_t row = 0; row < num_rows; row++) {
            for (uint32_t col = 0; col < num_cols; col++) {
                uint8_t pixel = bytes[i * (num_rows * num_cols) + row * num_cols + col];
                tensor_data(i, row, col) = ((float)pixel) / 255;
            }
        }
    }

    return true;
}

static bool ProcessMNISTLabels(const string& data, CPUTensor* out)
{
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(data.c_str());
    size_t len = data.size();

    uint32_t magic = swap(*(uint32_t*)bytes);
    FAIL_IF(magic != 0x00000801, "Invalid magic for MNIST labels file", false);
    bytes += 4;

    uint32_t num_labels = swap(*(uint32_t*)bytes);
    FAIL_IF(num_labels != len - 8, "Corrupted MNIST labels file, length mismatch.", false);
    bytes += 4;

    *out = CPUTensor({num_labels, 10}, ZeroInitializer);
    CPUTensor& tensor_data = *out;

    for (uint32_t i = 0; i < num_labels; i++) {
        uint8_t label = bytes[i];
        tensor_data(i, label) = 1.0;
    }

    return true;
}

bool LoadMNIST(std::string mnist_dir, nn::CPUTensor* train_data, nn::CPUTensor* train_labels, nn::CPUTensor* test_data, nn::CPUTensor* test_labels)
{
    string train_data_content, train_labels_content, test_data_content, test_labels_content;

    FAIL_IF(!LoadFile(mnist_dir + "/train-images-idx3-ubyte", &train_data_content), "MNIST Training images could not be loaded", false);
    FAIL_IF(!LoadFile(mnist_dir + "/train-labels-idx1-ubyte", &train_labels_content), "MNIST Training labels could not be loaded", false);
    FAIL_IF(!LoadFile(mnist_dir + "/t10k-images-idx3-ubyte", &test_data_content), "MNIST Test images could not be loaded", false);
    FAIL_IF(!LoadFile(mnist_dir + "/t10k-labels-idx1-ubyte", &test_labels_content), "MNIST Test lables could not be loaded", false);

    FAIL_IF(!ProcessMNISTData(train_data_content, train_data), "Could not process MNIST training data", false);
    FAIL_IF(!ProcessMNISTLabels(train_labels_content, train_labels), "Could not process MNIST training labels", false);
    FAIL_IF(!ProcessMNISTData(test_data_content, test_data), "Could not process MNIST test data", false);
    FAIL_IF(!ProcessMNISTLabels(test_labels_content, test_labels), "Could not process MNIST test labels", false);

    return true;
}

}       // namespace utils
