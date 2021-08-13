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

#include "includes.h"

/* constructor */
LayerInput::LayerInput(Params *params, int batchSize)
{
    prev_layer_ = NULL;

    type_ = "input";

    Assert(params->hasField("shape"), "Input Layer must have shape specified.");
    vector<int> shape = params->getVectori("shape");
    n_ = batchSize;
    c_ = shape[0];
    h_ = shape[1];
    w_ = shape[2];
    sample_size_ = c_ * h_ * w_;

    forwardTensor_ = new Tensor<float>(n_, c_, h_, w_);
    backwardTensor_ = new Tensor<float>(n_, c_, h_, w_);
}

/* destructor */
LayerInput::~LayerInput()
{
    delete forwardTensor_;
    delete backwardTensor_;
}

/* forward propagation */
void LayerInput::cpu_forward(int realBatchSize, bool train)
{
}

/* backward propagation */
void LayerInput::cpu_backward(int realBatchSize)
{
}

/* update weights and biases */
void LayerInput::cpu_update(int realBatchSize, float lr)
{
}

#if GPU == 1
/* forward propagation */
void LayerInput::gpu_forward(int realBatchSize, bool train)
{
}

/* backward propagation */
void LayerInput::gpu_backward(int realBatchSize)
{
}

/* update weights and biases */
void LayerInput::gpu_update(int realBatchSize, float lr)
{
}
#endif

/* initialize weights */
void LayerInput::initWeights(float *weights, int &offset)
{
}

/* get weights */
void LayerInput::getWeights(float *weights, int &offset)
{
}

/* get number of weights in this layer */
vector<int> LayerInput::getNumWeights()
{
    vector<int> num_weights{0};
    return num_weights;
}
