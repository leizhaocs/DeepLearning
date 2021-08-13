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

#ifndef _LAYER_BN_H__
#define _LAYER_BN_H__

#include "includes.h"

/* batch normalization layer */
class LayerBN : public Layer
{
public:
    /* constructor */
    LayerBN(Params *params, Layer *prev_layer);

    /* destructor */
    ~LayerBN();

    /* forward propagation */
    void cpu_forward(int realBatchSize, bool train);

    /* backward propagation */
    void cpu_backward(int realBatchSize);

    /* update weights and biases */
    void cpu_update(int realBatchSize, float lr);

#if GPU == 1
    /* forward propagation */
    void gpu_forward(int realBatchSize, bool train);

    /* backward propagation */
    void gpu_backward(int realBatchSize);

    /* update weights and biases */
    void gpu_update(int realBatchSize, float lr);
#endif

    /* initialize weights */
    void initWeights(float *weights, int &offset);

    /* get weights */
    void getWeights(float *weights, int &offset);

    /* get number of weights in this layer */
    std::vector<int> getNumWeights();

private:
    int channels_;                // the second dimension of input

    Tensor<float> *mean_;         // mean
    Tensor<float> *std_;          // std
    Tensor<float> *beta_;         // beta
    Tensor<float> *gamma_;        // gamma

    Tensor<float> *running_mean_; // running mean
    Tensor<float> *running_std_;  // running std
    Tensor<float> *grad_beta_;    // gradients of beta
    Tensor<float> *grad_gamma_;   // gradients of beta
};

#endif
