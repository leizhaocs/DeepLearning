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

#ifndef _LAYER_POOL_H__
#define _LAYER_POOL_H__

#include "includes.h"

class LayerPool : public Layer
{
public:
    /* constructor */
    LayerPool(Params *params, Layer *prev_layer);

    /* destructor */
    ~LayerPool();

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
    string pool_type_;            // pooling type

    int filter_h_;                // filter height
    int filter_w_;                // filter width

    int stride_h_;                // stride height
    int stride_w_;                // stride width

    int padding_h_;               // padding height
    int padding_w_;               // padding width

    Tensor<int> *indexTensor_;    // recording the index of the maximum neuron in the max pooling window
};

#endif
