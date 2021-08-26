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
LayerDrop::LayerDrop(Params *params, Layer *prev_layer)
{
    prev_layer_ = prev_layer;

    type_ = "dropout";

    n_ = prev_layer->n_;
    c_ = prev_layer->c_;
    h_ = prev_layer->h_;
    w_ = prev_layer->w_;
    sample_size_ = prev_layer->sample_size_;

    forwardTensor_ = new Tensor<float>(n_, c_, h_, w_);
    backwardTensor_ = new Tensor<float>(n_, c_, h_, w_);

    Assert(params->hasField("rate"), "Dropout Layer must have dropout rate specified.");
    rate_ = params->getScalarf("rate");

    dropoutTensor_ = new Tensor<float>(n_, c_, h_, w_);
}

/* destructor */
LayerDrop::~LayerDrop()
{
    delete forwardTensor_;
    delete backwardTensor_;
    delete dropoutTensor_;
}

/* forward propagation */
void LayerDrop::cpu_forward(int realBatchSize, bool train)
{
    clear(forwardTensor_);

    if (train)
    {
        random(dropoutTensor_);
        assign_cond(forwardTensor_, prev_layer_->forwardTensor_, dropoutTensor_, rate_);
    }
    else
    {
        assign(forwardTensor_, prev_layer_->forwardTensor_);
    }
}

/* backward propagation */
void LayerDrop::cpu_backward(int realBatchSize)
{
    clear(prev_layer_->backwardTensor_);

    assign_cond(prev_layer_->backwardTensor_, backwardTensor_, dropoutTensor_, rate_);
}

/* update weights and biases */
void LayerDrop::cpu_update(int realBatchSize, float lr)
{
}

#if GPU == 1
/* forward propagation */
void LayerDrop::gpu_forward(int realBatchSize, bool train)
{
    clear_gpu(forwardTensor_);

    if (train)
    {
        random_gpu(dropoutTensor_);
        assign_cond_gpu(forwardTensor_, prev_layer_->forwardTensor_, dropoutTensor_, rate_);
    }
    else
    {
        assign_gpu(forwardTensor_, prev_layer_->forwardTensor_);
    }
}

/* backward propagation */
void LayerDrop::gpu_backward(int realBatchSize)
{
    clear_gpu(prev_layer_->backwardTensor_);

    assign_cond_gpu(prev_layer_->backwardTensor_, backwardTensor_, dropoutTensor_, rate_);
}

/* update weights and biases */
void LayerDrop::gpu_update(int realBatchSize, float lr)
{
}
#endif

/* initialize weights */
void LayerDrop::initWeights(float *weights, int &offset)
{
}

/* get weights */
void LayerDrop::getWeights(float *weights, int &offset)
{
}

/* get number of weights in this layer */
vector<int> LayerDrop::getNumWeights()
{
    vector<int> num_weights{0};
    return num_weights;
}
