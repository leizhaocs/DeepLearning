# DeepLearning

## Introduction

This framework is written in C++ and CUDA. It supports the following features:

- Support both training and Inference.
- Support most common layer types in CNNs, e.g. conv layer, fully connected layer, pooling layer, activation layer, etc.
- Support running on both CPU and GPU.
- Support building the network structure by using a configuration file.
- Deep reinforcement learning (being implemented).

## Build

Simply type `make` to compile the code. If you want to compile the code without CUDA acceleration, change the first line in Makefile to:
```
GPU=0
```

## Run

This framework supports two running modes: *dnn* and *drl*.

### dnn mode

dnn mode is for conventional CNN training and testing. The command line for training is as following:
```
./learning dnn train <dataset name> <network cfg file> <load weights file[null]> <save weights file[null]> [-cpu]
```
The command line for testing is as following:
```
./learning dnn test  <dataset name> <network cfg file> <load weights file> [-cpu]
```
If `<load weights file>` is set to null, the network will randomly initialize the weights. If `<save weights file>` is set to null, the trained weights will not be saved. If `-cpu` is present, the framework will not use CUDA acceleration even if it is compiled with `GPU=1` in Makefile.

An example of training MNIST with CUDA acceleration is as following:
```
./learning dnn train mnist cfg/mnist_cnn.cfg null weights/mnist_cnn.weights
```
An example of test MNIST without CUDA acceleration is as following:
```
./learning dnn test mnist cfg/mnist_cnn.cfg weights/mnist_cnn.weights -cpu
```

### drl mode

drl mode is used for deep reinforcement learning. Currently, drl mode is under development. It is not fully functional yet.
