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
    clear(forwardTensor_->size(), forwardTensor_->getCpuPtr());

    for (int n = 0; n < realBatchSize; n++)
    {
        for (int c = 0; c < c_; c++)
        {
            for (int h = 0; h < h_; h++)
            {
                for (int w = 0; w < w_; w++)
                {
                    if (pool_type_ == "max")
                    {
                        int start_h = h * stride_h_ - padding_h_;
                        int start_w = w * stride_w_ - padding_w_;
                        int end_h = start_h + filter_h_;
                        int end_w = start_w + filter_w_;

                        float max = -FLT_MAX;
                        int max_i = -1;
                        for (int cur_h = start_h; cur_h < end_h; cur_h++)
                        {
                            for (int cur_w = start_w; cur_w < end_w; cur_w++)
                            {
                                if (cur_h < 0 || cur_w < 0 || cur_h > prev_layer_->h_-1 || cur_w > prev_layer_->w_-1)
                                {
                                }
                                else if (prev_layer_->forwardTensor_->data(n, c, cur_h, cur_w) > max)
                                {
                                    max = prev_layer_->forwardTensor_->data(n, c, cur_h, cur_w);
                                    max_i = ((n*prev_layer_->c_ + c)*prev_layer_->h_ + cur_h)*prev_layer_->w_ + cur_w;
                                }
                            }
                        }
                        forwardTensor_->data(n, c, h, w) = max;
                        indexTensor_->data(n, c, h, w) = max_i;
                    }
                    else
                    {
                        Assert(false, "Unrecognized pooling function.");
                    }
                }
            }
        }
    }
}

/* backward propagation */
void LayerPool::cpu_backward(int realBatchSize)
{
    if (prev_layer_->type_ == "input")
    {
        return;
    }

    clear(prev_layer_->backwardTensor_->size(), prev_layer_->backwardTensor_->getCpuPtr());

    if (pool_type_ == "max")
    {
        for (int i = 0; i < realBatchSize*sample_size_; ++i)
        {
            int index = indexTensor_->data(i);
            prev_layer_->backwardTensor_->data(index) += backwardTensor_->data(i);
        }
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
    clear_gpu(forwardTensor_->size(), forwardTensor_->getGpuPtr());

    if (pool_type_ == "max")
    {
        maxpool_gpu(prev_layer_->forwardTensor_->getGpuPtr(), forwardTensor_->getGpuPtr(), indexTensor_->getGpuPtr(),
            prev_layer_->h_, prev_layer_->w_, prev_layer_->c_, h_, w_,
            stride_h_, stride_w_, filter_h_, filter_w_, padding_h_, padding_w_, realBatchSize);
    }
    else
    {
        Assert(false, "Unrecognized pooling function.");
    }
}

/* backward propagation */
void LayerPool::gpu_backward(int realBatchSize)
{
    if (prev_layer_->type_ == "input")
    {
        return;
    }

    clear_gpu(prev_layer_->backwardTensor_->size(), prev_layer_->backwardTensor_->getGpuPtr());

    if (pool_type_ == "max")
    {
        backward_maxpool_gpu(backwardTensor_->getGpuPtr(), prev_layer_->backwardTensor_->getGpuPtr(), indexTensor_->getGpuPtr(),
            prev_layer_->h_, prev_layer_->w_, prev_layer_->c_, h_, w_,
            stride_h_, stride_w_, filter_h_, filter_w_, padding_h_, padding_w_, realBatchSize);
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
