#!/bin/bash
#
# Fetch the MNIST dataset.
#
# The code currently expects the data to be located in a directory called "mnist"
# at the top level directory. This script will set up everything correctly if run from
# that location.
#

mkdir -p mnist && cd mnist

wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

gunzip *gz
