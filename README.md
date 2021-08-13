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

dnn mode is to for conventional CNN training and testing. The command line for training is as following:
```
./learning dnn train <dataset name> <network cfg file> <load weights file[null]> <save weights file[null]> [-cpu]
```
  The command line for testing is as following:
  ```
  ./nn dnn test  <dataset name> <network cfg file> <load weights file>
  ```

### drl mode

drl mode is used for deep reinforcement learning. Currently, drl mode is under development. It is not fully functional yet.
