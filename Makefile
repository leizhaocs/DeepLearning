GPU=1
CUDNN=1

VPATH=./src/ ./src/data ./src/dnn ./src/drl ./src/utils ./src/lib
OBJDIR=./obj/
EXEC=learning

CC=gcc
CPP=g++

COMMON=-Isrc/ -Isrc/data/ -I/src/dnn -I/src/drl -Isrc/utils -Isrc/lib
CFLAGS=-fopenmp -O3 -std=c++11
LDFLAGS=-fopenmp

ifeq ($(GPU), 1)
NVCC=nvcc
ARCH=-gencode arch=compute_35,code=sm_35 \
     -gencode arch=compute_50,code=[sm_50,compute_50] \
     -gencode arch=compute_52,code=[sm_52,compute_52]
COMMON+=-DGPU=1 -I/usr/local/cuda/include/ -I/usr/local/cuda/NVIDIA_CUDA-11.3_Samples/common/inc/
NVCCFLAGS=-std=c++11
LDFLAGS+=-L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand -lstdc++ 
ifeq ($(CUDNN), 1)
COMMON+=-DCUDNN=1
LDFLAGS+=-lcudnn
endif
endif

SRC_CPP=$(notdir $(shell find src/ -type f -name '*.cpp'))
OBJ=$(SRC_CPP:.cpp=.o)
ifeq ($(GPU), 1)
SRC_CU=$(notdir $(shell find src/ -type f -name '*.cu'))
OBJ+=$(SRC_CU:.cu=.o)
endif

EXECOBJ=$(addprefix $(OBJDIR), $(OBJ))
DEPS=$(wildcard src/*.h) $(wildcard src/data/*.h) $(wildcard src/dnn/*.h) $(wildcard src/drl/*.h) $(wildcard src/lib/*.h) Makefile

all: obj $(EXEC)

$(EXEC): $(EXECOBJ)
	$(CPP) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CPP) $(COMMON) $(CFLAGS) -c $< -o $@

ifeq ($(GPU), 1)
$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(COMMON) $(NVCCFLAGS) -c $< -o $@
endif

obj:
	mkdir -p obj

.PHONY: clean clean_weights

clean:
	rm -rf $(EXEC) $(EXECOBJ) $(OBJDIR)/*

clean_weights:
	rm -rf weights/*
