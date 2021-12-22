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

#if GPU == 1
#if CUDNN == 1
    state_size_ = 0;
    states_ = NULL;
    reservespace_size_ = 0;
    reservespace_ = NULL;

    CHECK_CUDNN_ERRORS(cudnnCreateDropoutDescriptor(&dropout_desc_));
    CHECK_CUDNN_ERRORS(cudnnDropoutGetStatesSize(cudnn_handle(), &state_size_));
    if (state_size_ > 0)
    {
        if (states_ != NULL)
        {
            MemoryMonitor::instance()->freeGpuMemory(states_);
        }
        MemoryMonitor::instance()->gpuMalloc((void**)&states_, state_size_);
    }
    CHECK_CUDNN_ERRORS(cudnnSetDropoutDescriptor(dropout_desc_, cudnn_handle(), rate_, states_, state_size_, 0));
    CHECK_CUDNN_ERRORS(cudnnDropoutGetReserveSpaceSize(prev_layer_->forwardTensor_->getTensorDescriptor(), &reservespace_size_));
    if (reservespace_size_ > 0)
    {
        if (reservespace_ != NULL)
        {
            MemoryMonitor::instance()->freeGpuMemory(reservespace_);
        }
        MemoryMonitor::instance()->gpuMalloc((void**)&reservespace_, reservespace_size_);
    }
#endif
#endif
}

/* destructor */
LayerDrop::~LayerDrop()
{
    delete forwardTensor_;
    delete backwardTensor_;
    delete dropoutTensor_;
#if GPU == 1
#if CUDNN == 1
    CHECK_CUDNN_ERRORS(cudnnDestroyDropoutDescriptor(dropout_desc_));
    if (state_size_ > 0)
    {
        if (states_ != NULL)
        {
            MemoryMonitor::instance()->freeGpuMemory(states_);
        }
    }
    if (reservespace_size_ > 0)
    {
        if (reservespace_ != NULL)
        {
            MemoryMonitor::instance()->freeGpuMemory(reservespace_);
        }
    }
#endif
#endif
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

#if CUDNN == 1
    if (train)
    {
        CHECK_CUDNN_ERRORS(cudnnDropoutForward(cudnn_handle(), dropout_desc_,
            prev_layer_->forwardTensor_->getTensorDescriptor(), prev_layer_->forwardTensor_->getGpuPtr(),
            forwardTensor_->getTensorDescriptor(), forwardTensor_->getGpuPtr(),
            reservespace_, reservespace_size_));
    }
    else
    {
        cudaMemcpy(forwardTensor_->getGpuPtr(), prev_layer_->forwardTensor_->getGpuPtr(), forwardTensor_->total_bytes(), cudaMemcpyDeviceToDevice);
        CHECK_CUDA_ERRORS()
    }
#else
    if (train)
    {
        random_gpu(dropoutTensor_);
        assign_cond_gpu(forwardTensor_, prev_layer_->forwardTensor_, dropoutTensor_, rate_);
    }
    else
    {
        assign_gpu(forwardTensor_, prev_layer_->forwardTensor_);
    }
#endif
}

/* backward propagation */
void LayerDrop::gpu_backward(int realBatchSize)
{
    clear_gpu(prev_layer_->backwardTensor_);

#if CUDNN == 1
    CHECK_CUDNN_ERRORS(cudnnDropoutBackward(cudnn_handle(), dropout_desc_,
        backwardTensor_->getTensorDescriptor(), backwardTensor_->getGpuPtr(),
        prev_layer_->backwardTensor_->getTensorDescriptor(), prev_layer_->backwardTensor_->getGpuPtr(),
        reservespace_, reservespace_size_));
#else
    assign_cond_gpu(prev_layer_->backwardTensor_, backwardTensor_, dropoutTensor_, rate_);
#endif
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
