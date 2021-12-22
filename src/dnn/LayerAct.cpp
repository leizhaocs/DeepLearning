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
LayerAct::LayerAct(Params *params, Layer *prev_layer)
{
    prev_layer_ = prev_layer;

    type_ = "activation";

    n_ = prev_layer->n_;
    c_ = prev_layer->c_;
    h_ = prev_layer->h_;
    w_ = prev_layer->w_;
    sample_size_ = prev_layer->sample_size_;

    forwardTensor_ = new Tensor<float>(n_, c_, h_, w_);
    backwardTensor_ = new Tensor<float>(n_, c_, h_, w_);

    Assert(params->hasField("nonlinear"), "Activation Layer must have nonlinear specified.");
    nonlinear_ = params->getString("nonlinear");

#if GPU == 1
#if CUDNN == 1
    if (nonlinear_ == "relu")
    {
        CHECK_CUDNN_ERRORS(cudnnCreateActivationDescriptor(&act_desc_));
        CHECK_CUDNN_ERRORS(cudnnSetActivationDescriptor(act_desc_, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0));
    }
    else if (nonlinear_ == "sigmoid")
    {
        CHECK_CUDNN_ERRORS(cudnnCreateActivationDescriptor(&act_desc_));
        CHECK_CUDNN_ERRORS(cudnnSetActivationDescriptor(act_desc_, CUDNN_ACTIVATION_SIGMOID, CUDNN_PROPAGATE_NAN, 0));
    }
    else if (nonlinear_ == "tanh")
    {
        CHECK_CUDNN_ERRORS(cudnnCreateActivationDescriptor(&act_desc_));
        CHECK_CUDNN_ERRORS(cudnnSetActivationDescriptor(act_desc_, CUDNN_ACTIVATION_TANH, CUDNN_PROPAGATE_NAN, 0));
    }
    else if (nonlinear_ == "softmax")
    {
    }
    else
    {
        Assert(false, "Unsupported nonlinear function.");
    }
#endif
#endif
}

/* destructor */
LayerAct::~LayerAct()
{
    delete forwardTensor_;
    delete backwardTensor_;
#if GPU == 1
#if CUDNN == 1
    if ((nonlinear_ == "relu") || (nonlinear_ == "sigmoid") || (nonlinear_ == "tanh"))
    {
        CHECK_CUDNN_ERRORS(cudnnDestroyActivationDescriptor(act_desc_));
    }
#endif
#endif
}

/* forward propagation */
void LayerAct::cpu_forward(int realBatchSize, bool train)
{
    clear(forwardTensor_);

    if (nonlinear_ == "relu")
    {
        relu(prev_layer_->forwardTensor_, forwardTensor_);
    }
    else if (nonlinear_ == "sigmoid")
    {
        sigmoid(prev_layer_->forwardTensor_, forwardTensor_);
    }
    else if (nonlinear_ == "tanh")
    {
        tanh(prev_layer_->forwardTensor_, forwardTensor_);
    }
    else if (nonlinear_ == "softmax")
    {
        softmax(prev_layer_->forwardTensor_, forwardTensor_);
    }
    else
    {
        Assert(false, "Unsupported nonlinear function.");
    }
}

/* backward propagation */
void LayerAct::cpu_backward(int realBatchSize)
{
    clear(prev_layer_->backwardTensor_);

    if (nonlinear_ == "relu")
    {
        backward_relu(backwardTensor_, forwardTensor_, prev_layer_->backwardTensor_);
    }
    else if (nonlinear_ == "sigmoid")
    {
        backward_sigmoid(backwardTensor_, forwardTensor_, prev_layer_->backwardTensor_);
    }
    else if (nonlinear_ == "tanh")
    {
        backward_tanh(backwardTensor_, forwardTensor_, prev_layer_->backwardTensor_);
    }
    else if (nonlinear_ == "softmax")
    {
        backward_softmax(backwardTensor_, forwardTensor_, prev_layer_->backwardTensor_);
    }
    else
    {
        Assert(false, "Unsupported nonlinear function.");
    }
}

/* update weights and biases */
void LayerAct::cpu_update(int realBatchSize, float lr)
{
}

#if GPU == 1
/* forward propagation */
void LayerAct::gpu_forward(int realBatchSize, bool train)
{
    clear_gpu(forwardTensor_);

#if CUDNN == 1
    const float one  =  1.f;
    const float zero =  0.f;

    if ((nonlinear_ == "relu") || (nonlinear_ == "sigmoid") || (nonlinear_ == "tanh"))
    {
        CHECK_CUDNN_ERRORS(cudnnActivationForward(cudnn_handle(), act_desc_, &one,
            prev_layer_->forwardTensor_->getTensorDescriptor(), prev_layer_->forwardTensor_->getGpuPtr(),
            &zero,
            forwardTensor_->getTensorDescriptor(), forwardTensor_->getGpuPtr()));
    }
    else if (nonlinear_ == "softmax")
    {
        CHECK_CUDNN_ERRORS(cudnnSoftmaxForward(cudnn_handle(), CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &one,
            prev_layer_->forwardTensor_->getTensorDescriptor(), prev_layer_->forwardTensor_->getGpuPtr(),
            &zero,
            forwardTensor_->getTensorDescriptor(), forwardTensor_->getGpuPtr()));
    }
    else
    {
        Assert(false, "Unsupported nonlinear function.");
    }
#else
    if (nonlinear_ == "relu")
    {
        relu_gpu(prev_layer_->forwardTensor_, forwardTensor_);
    }
    else if (nonlinear_ == "sigmoid")
    {
        sigmoid_gpu(prev_layer_->forwardTensor_, forwardTensor_);
    }
    else if (nonlinear_ == "tanh")
    {
        tanh_gpu(prev_layer_->forwardTensor_, forwardTensor_);
    }
    else if (nonlinear_ == "softmax")
    {
        softmax_gpu(prev_layer_->forwardTensor_, forwardTensor_);
    }
    else
    {
        Assert(false, "Unsupported nonlinear function.");
    }
#endif
}

/* backward propagation */
void LayerAct::gpu_backward(int realBatchSize)
{
    clear_gpu(prev_layer_->backwardTensor_);

#if CUDNN == 1
    const float one  =  1.f;
    const float zero =  0.f;

    if ((nonlinear_ == "relu") || (nonlinear_ == "sigmoid") || (nonlinear_ == "tanh"))
    {
        CHECK_CUDNN_ERRORS(cudnnActivationBackward(cudnn_handle(), act_desc_, &one,
            forwardTensor_->getTensorDescriptor(), forwardTensor_->getGpuPtr(),
            backwardTensor_->getTensorDescriptor(), backwardTensor_->getGpuPtr(),
            prev_layer_->forwardTensor_->getTensorDescriptor(), prev_layer_->forwardTensor_->getGpuPtr(),
            &zero,
            prev_layer_->backwardTensor_->getTensorDescriptor(), prev_layer_->backwardTensor_->getGpuPtr()));
    }
    else if (nonlinear_ == "softmax")
    {
        CHECK_CUDNN_ERRORS(cudnnSoftmaxBackward(cudnn_handle(), CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &one,
            forwardTensor_->getTensorDescriptor(), forwardTensor_->getGpuPtr(),
            backwardTensor_->getTensorDescriptor(), backwardTensor_->getGpuPtr(),
            &zero,
            prev_layer_->backwardTensor_->getTensorDescriptor(), prev_layer_->backwardTensor_->getGpuPtr()));
    }
    else
    {
        Assert(false, "Unsupported nonlinear function.");
    }
#else
    if (nonlinear_ == "relu")
    {
        backward_relu_gpu(backwardTensor_, forwardTensor_, prev_layer_->backwardTensor_);
    }
    else if (nonlinear_ == "sigmoid")
    {
        backward_sigmoid_gpu(backwardTensor_, forwardTensor_, prev_layer_->backwardTensor_);
    }
    else if (nonlinear_ == "tanh")
    {
        backward_tanh_gpu(backwardTensor_, forwardTensor_, prev_layer_->backwardTensor_);
    }
    else if (nonlinear_ == "softmax")
    {
        backward_softmax_gpu(backwardTensor_, forwardTensor_, prev_layer_->backwardTensor_);
    }
    else
    {
        Assert(false, "Unsupported nonlinear function.");
    }
#endif
}

/* update weights and biases */
void LayerAct::gpu_update(int realBatchSize, float lr)
{
}
#endif

/* initialize weights */
void LayerAct::initWeights(float *weights, int &offset)
{
}

/* get weights */
void LayerAct::getWeights(float *weights, int &offset)
{
}

/* get number of weights in this layer */
vector<int> LayerAct::getNumWeights()
{
    vector<int> num_weights{0};
    return num_weights;
}
