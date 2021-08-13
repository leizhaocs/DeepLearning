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
