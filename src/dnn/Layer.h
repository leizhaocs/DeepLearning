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

#ifndef _LAYER_H__
#define _LAYER_H__

#include "includes.h"

/* layer base class */
class Layer
{
public:
    /* constructor */
    Layer();

    /* destructor */
    virtual ~Layer();

    /* forward propagation */
    virtual void cpu_forward(int realBatchSize, bool train) = 0;

    /* backward propagation */
    virtual void cpu_backward(int realBatchSize) = 0;

    /* update weights and biases */
    virtual void cpu_update(int realBatchSize, float lr) = 0;

#if GPU == 1
    /* forward propagation */
    virtual void gpu_forward(int realBatchSize, bool train) = 0;

    /* backward propagation */
    virtual void gpu_backward(int realBatchSize) = 0;

    /* update weights and biases */
    virtual void gpu_update(int realBatchSize, float lr) = 0;
#endif

    /* initialize weights */
    virtual void initWeights(float *weights, int &offset) = 0;

    /* get weights */
    virtual void getWeights(float *weights, int &offset) = 0;

    /* get number of weights in this layer */
    virtual vector<int> getNumWeights() = 0;

    Layer *prev_layer_;                     // previous layer
    string type_;                           // layer type
    int n_;                                 // output's batch size
    int c_;                                 // output's channel
    int h_;                                 // output's height
    int w_;                                 // output's width
    int sample_size_;                       // c_*h_*w_
    Tensor<float> *forwardTensor_;          // output of this layer in forward propagation (n,h,w,c)
    Tensor<float> *backwardTensor_;         // input of this layer in backward propagration (n,h,w,c)
};

#endif
