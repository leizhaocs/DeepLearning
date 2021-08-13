/**********************************************************************
 *
 * Copyright Lei Zhao.
 * contact: leizhao0403@gmail.com
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 **********************************************************************/

#ifndef _DEFINES_H_
#define _DEFINES_H_

#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <random>
#include <algorithm>
#include <map>
#include <string>
#include <chrono>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>
#include <float.h>
#if GPU == 1
#include <cublas_v2.h>
#include <curand.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <helper_cuda.h>
#endif

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

///////////////////////////////////////////////
///////////////////////////////////////////////
///////////////////////////////////////////////

using namespace std;

extern bool use_gpu;

#define PORT 65432
#define OPENMP_THREADS 8
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#define EPSILON 1e-4
#if GPU == 1
#define BLOCK 512
#endif

class MemoryMoniter;
template<typename T>
class Tensor;
class Params;
class Layer;
class LayerInput;
class LayerAct;
class LayerConv;
class LayerFull;
class LayerPool;
class LayerBN;
class LayerDrop;
class Net;
class ReplayMemory;
class Transition;
class Action;
class Observation;
class Agent;

/* assert */
inline void Assert(bool b, string msg)
{
    if (b)
    {
        return;
    }
    string _errmsg = string("Assertion Failed: ") + msg;
    cerr << _errmsg.c_str() << endl;
    exit(1);
}

#if GPU == 1
#include "utils/cuda_util.h"
#endif
#include "utils/args.h"
#include "utils/config.h"
#include "utils/gemm.h"
#include "utils/blas.h"
#include "utils/conv.h"
#include "utils/gym.h"
#include "utils/loss.h"
#include "utils/MemoryMonitor.h"
#include "utils/Tensor.h"
#include "utils/Params.h"
#include "utils/weights.h"
#include "dnn/Layer.h"
#include "dnn/LayerInput.h"
#include "dnn/LayerAct.h"
#include "dnn/LayerConv.h"
#include "dnn/LayerFull.h"
#include "dnn/LayerPool.h"
#include "dnn/LayerBN.h"
#include "dnn/LayerDrop.h"
#include "dnn/Net.h"
#include "drl/ReplayMemory.h"
#include "drl/Transition.h"
#include "drl/Action.h"
#include "drl/Observation.h"
#include "drl/Agent.h"

#endif
