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

#ifndef _LAYER_FULL_H__
#define _LAYER_FULL_H__

#include "includes.h"

/* fully connected layer */
class LayerFull : public Layer
{
public:
    /* constructor */
    LayerFull(Params *params, Layer *prev_layer);

    /* destructor */
    ~LayerFull();

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
    vector<int> getNumWeights();

private:
    Tensor<float> *weights_;        // weights (prev_layer->sample_size_, sample_size_)
    Tensor<float> *biases_;         // biases (sample_size_)

    Tensor<float> *grad_weights_;   // gradients of weights (prev_layer->sample_size_, sample_size_)
    Tensor<float> *grad_biases_;    // gradients of biases (sample_size_)

#if GPU == 1
#if CUDNN == 1
    Tensor<float> *one_vec_;        // a vector of all 1s, used for adding biases to output
#endif
#endif
};

#endif
