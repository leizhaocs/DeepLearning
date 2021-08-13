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

#ifndef _NET_H__
#define _NET_H__

#include "includes.h"

/* neural network */
class Net
{
public:
    /* constructor */
    Net(Params *layer_params, int layers_num, int batchSize, string loss_func);

    /* destructor */
    ~Net();

    /* initialize weights for all layers */
    void initWeights(float *weights);

    /* get weights from all layers */
    void getWeights(float *weights);

    /* get number of weights per layer, only return layers with weights, 0: real 1: binary */
    int getNumWeights(vector<vector<int>> *total_weights);

    /* init the input of the first layer, (channels, height, width) match the dimension of first layer */
    void initForward(Tensor<float> *inputs, int inputs_offset, int forwardTensor_offset, int realBatchSize, int channels, int height, int width);

    /* forward propagate through all layers */
    void forward(int realBatchSize, bool train);

    /* calculate the loss before backward propagation */
    void loss(Tensor<float> *targets, int targets_offset, int realBatchSize, int classes);

    /* backward propagate through all layers */
    void backward(int realBatchSize);

    /* update weights and biases */
    void update(int realBatchSize, float lr);

    /* get the outputs of the final layer, classes must match last layer */
    void getOutputs(Tensor<float> *outputs, int outputs_offset, int forwardTensor_offset, int realBatchSize, int classes);

    int num_layers_;               // number of layers
    vector<Layer *> layers_;       // all the layers
    string loss_;                  // loss function
};

#endif
