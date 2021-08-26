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
LayerPool::LayerPool(Params *params, Layer *prev_layer)
{
    prev_layer_ = prev_layer;

    type_ = "pool";

    Assert(params->hasField("poolType"), "Pool Layer must have pool type specified.");
    pool_type_ = params->getString("poolType");

    Assert(params->hasField("filterSize"), "Pool Layer must have filter size specified.");
    vector<int> filterSize = params->getVectori("filterSize");
    filter_h_ = filterSize[0];
    filter_w_ = filterSize[1];

    Assert(params->hasField("stride"), "Pool Layer must have stride specified.");
    vector<int> stride = params->getVectori("stride");
    stride_h_ = stride[0];
    stride_w_ = stride[1];

    Assert(params->hasField("padding"), "Pool Layer must have padding specified.");
    vector<int> padding = params->getVectori("padding");
    padding_h_ = padding[0];
    padding_w_ = padding[1];

    n_ = prev_layer->n_;
    c_ = prev_layer->c_;
    h_ = (prev_layer_->h_-filter_h_+2*padding_h_)/stride_h_ + 1;
    w_ = (prev_layer_->w_-filter_w_+2*padding_w_)/stride_w_ + 1;
    sample_size_ = c_ * h_ * w_;

    forwardTensor_ = new Tensor<float>(n_, c_, h_, w_);
    backwardTensor_ = new Tensor<float>(n_, c_, h_, w_);

    indexTensor_ = new Tensor<int>(n_, c_, h_, w_);
}

/* destructor */
LayerPool::~LayerPool()
{
    delete forwardTensor_;
    delete backwardTensor_;
    delete indexTensor_;
}

/* forward propagation */
void LayerPool::cpu_forward(int realBatchSize, bool train)
{
    clear(forwardTensor_);

    if (pool_type_ == "max")
    {
        maxpool(prev_layer_->forwardTensor_, forwardTensor_, indexTensor_, stride_h_, stride_w_, filter_h_, filter_w_, padding_h_, padding_w_);
    }
    else
    {
        Assert(false, "Unrecognized pooling function.");
    }
}

/* backward propagation */
void LayerPool::cpu_backward(int realBatchSize)
{
    clear(prev_layer_->backwardTensor_);

    if (pool_type_ == "max")
    {
        backward_maxpool(backwardTensor_, prev_layer_->backwardTensor_, indexTensor_);
    }
    else
    {
        Assert(false, "Unrecognized pooling function.");
    }
}

/* update weights and biases */
void LayerPool::cpu_update(int realBatchSize, float lr)
{
}

#if GPU == 1
/* forward propagation */
void LayerPool::gpu_forward(int realBatchSize, bool train)
{
    clear_gpu(forwardTensor_);

    if (pool_type_ == "max")
    {
        maxpool_gpu(prev_layer_->forwardTensor_, forwardTensor_, indexTensor_, stride_h_, stride_w_, filter_h_, filter_w_, padding_h_, padding_w_);
    }
    else
    {
        Assert(false, "Unrecognized pooling function.");
    }
}

/* backward propagation */
void LayerPool::gpu_backward(int realBatchSize)
{
    clear_gpu(prev_layer_->backwardTensor_);

    if (pool_type_ == "max")
    {
        backward_maxpool_gpu(backwardTensor_, prev_layer_->backwardTensor_, indexTensor_, stride_h_, stride_w_, filter_h_, filter_w_, padding_h_, padding_w_);
    }
    else
    {
        Assert(false, "Unrecognized pooling function.");
    }
}

/* update weights and biases */
void LayerPool::gpu_update(int realBatchSize, float lr)
{
}
#endif

/* initialize weights */
void LayerPool::initWeights(float *weights, int &offset)
{
}

/* get weights */
void LayerPool::getWeights(float *weights, int &offset)
{
}

/* get number of weights in this layer */
vector<int> LayerPool::getNumWeights()
{
    vector<int> num_weights{0};
    return num_weights;
}
