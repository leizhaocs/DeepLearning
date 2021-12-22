# DeepLearning

## Introduction

Initially, I wrote this framework based on the needs of my research projects, which use custom computation operators to emulate the DNN execution on ASIC accelerators. For example, in order to simulate an accelerator that uses Stochastic Computation (SC) or Homomorphic Encryption (HE), one needs to replace all the add and multiply operations to SC operations or HE operations. The best way to implement this is to overload the original + and * (and +=, -=, ect., you can imagine the huge amount of work if you examine the whole code base of a large framework like PyTorch or Tensorflow) operators in C++. However, doing this in the commonly used large frameworks (such as PyTorch or Tensorflow) is not an easy task because of their massive code size. So I decided to wirte my own framework. I handcrafted all the code for both C++ and CUDA to make the operator overloading task less pain. So, currently the cuda kernels only provide the correct functionality, fully optimizations are still left for future work. Later, I decided to make this framework more useful for general purposes, so I also implemented the CuDNN implementation to acheive comparable execution speed with other frameworks. If you are not going to modify the lowest level implementation as I did in my own research project, this framework also suites most of the general machine learning purposes by enabling CuDNN support. I have included comments with a breif description for almost all functions in the code, hopefully extending this framework based on your own needs will not be a hard problem.

This framework is written in C++ and CUDA. It supports the following features:

- Support both training and Inference.
- Support most common layer types in CNNs, e.g. conv layer, fully connected layer, pooling layer, activation layer, etc.
- Support running on both CPU and GPU.
- Support acceleration with CuDNN.
- Support building the network structure by using a configuration file.
- Deep reinforcement learning (being implemented).

More details of the framework and the underlying implementation details are on https://leizhaocs.github.io/DeepLearning/.

## Build

Simply type `make` to compile the code. The default is GPU enabled with cudnn to accelerate most of the layers.
If you want to compile the code without CUDA acceleration, change the first line in Makefile to:
```
GPU=0
```
If you want to use my handcrafted CUDA implementation (not fully optimized, so much slower) instead if CuDNN, change the second line in Makefile to:
```
CUDNN=0
```
Note that CuDNN is only enabled when both `GPU` and `CUDNN` are set to 1.

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
