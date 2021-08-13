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
}

/* destructor */
LayerAct::~LayerAct()
{
    delete forwardTensor_;
    delete backwardTensor_;
}

/* forward propagation */
void LayerAct::cpu_forward(int realBatchSize, bool train)
{
    clear(forwardTensor_->size(), forwardTensor_->getCpuPtr());

    if (nonlinear_ == "relu")
    {
        for (int i = 0; i < forwardTensor_->size(); i++)
        {
            if (prev_layer_->forwardTensor_->data(i) > 0)
            {
                forwardTensor_->data(i) = prev_layer_->forwardTensor_->data(i);
            }
            else
            {
                forwardTensor_->data(i) = 0;
            }
        }
    }
    else if (nonlinear_ == "sigmoid")
    {
        for (int i = 0; i < forwardTensor_->size(); i++)
        {
            forwardTensor_->data(i) = exp((float)prev_layer_->forwardTensor_->data(i));
            forwardTensor_->data(i) = forwardTensor_->data(i) / (forwardTensor_->data(i) + 1);
        }
    }
    else if (nonlinear_ == "softmax")
    {
        for (int n = 0; n < realBatchSize; n++)
        {
            float max = -FLT_MAX;
            for (int i = 0; i < sample_size_; i++)
            {
                if (prev_layer_->forwardTensor_->data(n, i) > max)
                {
                    max = prev_layer_->forwardTensor_->data(n, i);
                }
            }
            float sum = 0;
            for (int i = 0; i < sample_size_; i++)
            {
                forwardTensor_->data(n, i) = exp((float)(prev_layer_->forwardTensor_->data(n, i)-max));
                sum += forwardTensor_->data(n, i);
            }
            for (int i = 0; i < sample_size_; i++)
            {
                forwardTensor_->data(n, i) /= sum;
            }
        }
    }
    else
    {
        Assert(false, "Unsupported nonlinear function.");
    }
}

/* backward propagation */
void LayerAct::cpu_backward(int realBatchSize)
{
    if (prev_layer_->type_ == "input")
    {
        return;
    }

    clear(prev_layer_->backwardTensor_->size(), prev_layer_->backwardTensor_->getCpuPtr());

    if (nonlinear_ == "relu")
    {
        for (int i = 0; i < prev_layer_->backwardTensor_->size(); i++)
        {
            if (prev_layer_->forwardTensor_->data(i) > 0)
            {
                prev_layer_->backwardTensor_->data(i) = backwardTensor_->data(i);
            }
            else
            {
                prev_layer_->backwardTensor_->data(i) = 0;
            }
        }
    }
    else if (nonlinear_ == "sigmoid")
    {
        for (int i = 0; i < prev_layer_->backwardTensor_->size(); i++)
        {
            prev_layer_->backwardTensor_->data(i) = backwardTensor_->data(i) * forwardTensor_->data(i) * (1 - forwardTensor_->data(i));
        }
    }
    else if (nonlinear_ == "softmax")
    {
        for (int n = 0; n < realBatchSize; n++)
        {
            for (int i = 0; i < sample_size_; i++)
            {
                if (backwardTensor_->data(n, i) != 0)
                {
                    prev_layer_->backwardTensor_->data(n, i) = forwardTensor_->data(n, i) - 1;
                }
                else
                {
                    prev_layer_->backwardTensor_->data(n, i) = forwardTensor_->data(n, i);
                }
            }
        }
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
    clear_gpu(forwardTensor_->size(), forwardTensor_->getGpuPtr());

    if (nonlinear_ == "relu")
    {
        relu_gpu(prev_layer_->forwardTensor_->getGpuPtr(), forwardTensor_->getGpuPtr(), realBatchSize*sample_size_);
    }
    else if (nonlinear_ == "sigmoid")
    {
        sigmoid_gpu(prev_layer_->forwardTensor_->getGpuPtr(), forwardTensor_->getGpuPtr(), realBatchSize*sample_size_);
    }
    else if (nonlinear_ == "softmax")
    {
        softmax_gpu(prev_layer_->forwardTensor_->getGpuPtr(), forwardTensor_->getGpuPtr(), sample_size_, realBatchSize);
    }
    else
    {
        Assert(false, "Unsupported nonlinear function.");
    }
}

/* backward propagation */
void LayerAct::gpu_backward(int realBatchSize)
{
    if (prev_layer_->type_ == "input")
    {
        return;
    }

    clear_gpu(prev_layer_->backwardTensor_->size(), prev_layer_->backwardTensor_->getGpuPtr());

    if (nonlinear_ == "relu")
    {
        backward_relu_gpu(backwardTensor_->getGpuPtr(), forwardTensor_->getGpuPtr(),
            prev_layer_->backwardTensor_->getGpuPtr(), realBatchSize*sample_size_);
    }
    else if (nonlinear_ == "sigmoid")
    {
        backward_sigmoid_gpu(backwardTensor_->getGpuPtr(), forwardTensor_->getGpuPtr(),
            prev_layer_->backwardTensor_->getGpuPtr(), realBatchSize*sample_size_);
    }
    else if (nonlinear_ == "softmax")
    {
        backward_softmax_gpu(backwardTensor_->getGpuPtr(), forwardTensor_->getGpuPtr(),
            prev_layer_->backwardTensor_->getGpuPtr(), sample_size_, realBatchSize);
    }
    else
    {
        Assert(false, "Unsupported nonlinear function.");
    }
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
