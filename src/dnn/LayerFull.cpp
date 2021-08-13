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
LayerFull::LayerFull(Params *params, Layer *prev_layer)
{
    prev_layer_ = prev_layer;

    type_ = "full";

    Assert(params->hasField("length"), "Full Layer must have length specified.");
    c_ = params->getScalari("length");
    n_ = prev_layer->n_;
    h_ = 1;
    w_ = 1;
    sample_size_ = c_;

    forwardTensor_ = new Tensor<float>(n_, c_, h_, w_);
    backwardTensor_ = new Tensor<float>(n_, c_, h_, w_);

    weights_ = new Tensor<float>(sample_size_, 1, 1, prev_layer->sample_size_);
    biases_ = new Tensor<float>(1, 1, 1, sample_size_);

    grad_weights_ = new Tensor<float>(sample_size_, 1, 1, prev_layer->sample_size_);
    grad_biases_ = new Tensor<float>(1, 1, 1, sample_size_);
}

/* destructor */
LayerFull::~LayerFull()
{
    delete forwardTensor_;
    delete backwardTensor_;
    delete weights_;
    delete biases_;
    delete grad_weights_;
    delete grad_biases_;
}

/* forward propagation */
void LayerFull::cpu_forward(int realBatchSize, bool train)
{
    clear(forwardTensor_->size(), forwardTensor_->getCpuPtr());

    int m = realBatchSize;
    int n = sample_size_;
    int k = prev_layer_->sample_size_;

    float *a = prev_layer_->forwardTensor_->getCpuPtr();
    float *b = weights_->getCpuPtr();
    float *c = forwardTensor_->getCpuPtr();

    gemm(0, 1, m, n, k, a, k, b, k, c, n);
    add_bias(forwardTensor_->getCpuPtr(), biases_->getCpuPtr(), realBatchSize, sample_size_, 1);
}

/* backward propagation */
void LayerFull::cpu_backward(int realBatchSize)
{
    if (prev_layer_->type_ == "input")
    {
        return;
    }

    clear(prev_layer_->backwardTensor_->size(), prev_layer_->backwardTensor_->getCpuPtr());
    clear(grad_weights_->size(), grad_weights_->getCpuPtr());
    clear(grad_biases_->size(), grad_biases_->getCpuPtr());

    int m = realBatchSize;
    int n = prev_layer_->sample_size_;
    int k = sample_size_;

    float *a = backwardTensor_->getCpuPtr();
    float *b = weights_->getCpuPtr();
    float *c = prev_layer_->backwardTensor_->getCpuPtr();

    gemm(0, 0, m, n, k, a, k, b, n, c, n);

    m = sample_size_;
    n = prev_layer_->sample_size_;
    k = realBatchSize;

    a = backwardTensor_->getCpuPtr();
    b = prev_layer_->forwardTensor_->getCpuPtr();
    c = grad_weights_->getCpuPtr();

    gemm(1, 0, m, n, k, a, m, b, n, c, n);
    backward_bias(grad_biases_->getCpuPtr(), backwardTensor_->getCpuPtr(), realBatchSize, sample_size_, 1);
}

/* update weights and biases */
void LayerFull::cpu_update(int realBatchSize, float lr)
{
    float step = -lr/realBatchSize;
    axpy(weights_->size(), step, grad_weights_->getCpuPtr(), weights_->getCpuPtr());
    axpy(biases_->size(), step, grad_biases_->getCpuPtr(), biases_->getCpuPtr());
}

#if GPU == 1
/* forward propagation */
void LayerFull::gpu_forward(int realBatchSize, bool train)
{
    clear_gpu(forwardTensor_->size(), forwardTensor_->getGpuPtr());

    int m = realBatchSize;
    int n = sample_size_;
    int k = prev_layer_->sample_size_;

    float *a = prev_layer_->forwardTensor_->getGpuPtr();
    float *b = weights_->getGpuPtr();
    float *c = forwardTensor_->getGpuPtr();

    gemm_gpu(0, 1, m, n, k, a, k, b, k, c, n);
    add_bias_gpu(forwardTensor_->getGpuPtr(), biases_->getGpuPtr(), realBatchSize, sample_size_, 1);
}

/* backward propagation */
void LayerFull::gpu_backward(int realBatchSize)
{
    if (prev_layer_->type_ == "input")
    {
        return;
    }

    clear_gpu(prev_layer_->backwardTensor_->size(), prev_layer_->backwardTensor_->getGpuPtr());
    clear_gpu(grad_weights_->size(), grad_weights_->getGpuPtr());
    clear_gpu(grad_biases_->size(), grad_biases_->getGpuPtr());

    int m = realBatchSize;
    int n = prev_layer_->sample_size_;
    int k = sample_size_;

    float *a = backwardTensor_->getGpuPtr();
    float *b = weights_->getGpuPtr();
    float *c = prev_layer_->backwardTensor_->getGpuPtr();

    gemm_gpu(0, 0, m, n, k, a, k, b, n, c, n);

    m = sample_size_;
    n = prev_layer_->sample_size_;
    k = realBatchSize;

    a = backwardTensor_->getGpuPtr();
    b = prev_layer_->forwardTensor_->getGpuPtr();
    c = grad_weights_->getGpuPtr();

    gemm_gpu(1, 0, m, n, k, a, m, b, n, c, n);
    backward_bias_gpu(grad_biases_->getGpuPtr(), backwardTensor_->getGpuPtr(), realBatchSize, sample_size_, 1);
}

/* update weights and biases */
void LayerFull::gpu_update(int realBatchSize, float lr)
{
    float step = -lr/realBatchSize;
    axpy_gpu(weights_->size(), step, grad_weights_->getGpuPtr(), weights_->getGpuPtr());
    axpy_gpu(biases_->size(), step, grad_biases_->getGpuPtr(), biases_->getGpuPtr());
}
#endif

/* initialize weights */
void LayerFull::initWeights(float *weights, int &offset)
{
    for (int i = 0; i < weights_->size(); i++)
    {
        weights_->data(i) = weights[offset+i];
    }
    offset += weights_->size();

    for (int i = 0; i < biases_->size(); i++)
    {
        biases_->data(i) = weights[offset+i];
    }
    offset += biases_->size();

#if GPU == 1
    if (use_gpu)
    {
        weights_->toGpu();
        biases_->toGpu();
    }
#endif
}

/* get weights */
void LayerFull::getWeights(float *weights, int &offset)
{
#if GPU == 1
    if (use_gpu)
    {
        weights_->toCpu();
        biases_->toCpu();
    }
#endif

    for (int i = 0; i < weights_->size(); i++)
    {
        weights[offset+i] = weights_->data(i);
    }
    offset += weights_->size();

    for (int i = 0; i < biases_->size(); i++)
    {
        weights[offset+i] = biases_->data(i);
    }
    offset += biases_->size();
}

/* get number of weights in this layer */
vector<int> LayerFull::getNumWeights()
{
    int nw = weights_->size();
    int nb = biases_->size();
    vector<int> num_weights;
    num_weights.push_back(nw+nb);
    num_weights.push_back(nw);
    num_weights.push_back(nb);
    return num_weights;
}
