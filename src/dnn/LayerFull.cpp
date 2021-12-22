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

#if GPU == 1
#if CUDNN == 1
    one_vec_ = new Tensor<float>(1, n_, 1, 1);
    for (int i = 0; i < n_; i++)
    {
        one_vec_->data(i) = 1;
    }
    one_vec_->toGpu();
#endif
#endif

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
    clear(forwardTensor_);

    int m = realBatchSize;
    int n = sample_size_;
    int k = prev_layer_->sample_size_;

    float *a = prev_layer_->forwardTensor_->getCpuPtr();
    float *b = weights_->getCpuPtr();
    float *c = forwardTensor_->getCpuPtr();

    gemm(0, 1, m, n, k, a, k, b, k, c, n);
    add_expand_channel(forwardTensor_, biases_);
}

/* backward propagation */
void LayerFull::cpu_backward(int realBatchSize)
{
    clear(prev_layer_->backwardTensor_);
    clear(grad_weights_);
    clear(grad_biases_);

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
    backward_bias(grad_biases_, backwardTensor_);
}

/* update weights and biases */
void LayerFull::cpu_update(int realBatchSize, float lr)
{
    float step = -lr/realBatchSize;
    axpy(weights_, grad_weights_, step);
    axpy(biases_, grad_biases_, step);
}

#if GPU == 1
/* forward propagation */
void LayerFull::gpu_forward(int realBatchSize, bool train)
{
    clear_gpu(forwardTensor_);

#if CUDNN == 1
    const float one  =  1.f;
    const float zero =  0.f;

    CHECK_CUBLAS_ERRORS(cublasSgemm(cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, 
        sample_size_, realBatchSize, prev_layer_->sample_size_,
        &one,  
        weights_->getGpuPtr(), prev_layer_->sample_size_, 
        prev_layer_->forwardTensor_->getGpuPtr(), prev_layer_->sample_size_,
        &zero, 
        forwardTensor_->getGpuPtr(),  sample_size_));

    CHECK_CUBLAS_ERRORS(cublasSgemm(cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, 
        sample_size_, realBatchSize, 1,
        &one, 
        biases_->getGpuPtr(), sample_size_, 
        one_vec_->getGpuPtr(), 1, &one, 
        forwardTensor_->getGpuPtr(),  sample_size_));
#else
    int m = realBatchSize;
    int n = sample_size_;
    int k = prev_layer_->sample_size_;

    float *a = prev_layer_->forwardTensor_->getGpuPtr();
    float *b = weights_->getGpuPtr();
    float *c = forwardTensor_->getGpuPtr();

    gemm_gpu(0, 1, m, n, k, a, k, b, k, c, n);
    add_expand_channel_gpu(forwardTensor_, biases_);
#endif
}

/* backward propagation */
void LayerFull::gpu_backward(int realBatchSize)
{
    clear_gpu(prev_layer_->backwardTensor_);
    clear_gpu(grad_weights_);
    clear_gpu(grad_biases_);

#if CUDNN == 1
    const float one  =  1.f;
    const float zero =  0.f;

    CHECK_CUBLAS_ERRORS(cublasSgemv(cublas_handle(), CUBLAS_OP_N,
        sample_size_, realBatchSize,
        &one,
        backwardTensor_->getGpuPtr(), sample_size_,
        one_vec_->getGpuPtr(), 1, &zero,
        grad_biases_->getGpuPtr(), 1));

    CHECK_CUBLAS_ERRORS(cublasSgemm(cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T,
        prev_layer_->sample_size_, sample_size_, realBatchSize,
        &one,
        prev_layer_->forwardTensor_->getGpuPtr(), prev_layer_->sample_size_,
        backwardTensor_->getGpuPtr(), sample_size_,
        &zero,
        grad_weights_->getGpuPtr(), prev_layer_->sample_size_));

    CHECK_CUBLAS_ERRORS(cublasSgemm(cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
        prev_layer_->sample_size_, realBatchSize, sample_size_,
        &one,
        weights_->getGpuPtr(), prev_layer_->sample_size_,
        backwardTensor_->getGpuPtr(), sample_size_,
        &zero, 
        prev_layer_->backwardTensor_->getGpuPtr(), prev_layer_->sample_size_));
#else
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
    backward_bias_gpu(grad_biases_, backwardTensor_);
#endif
}

/* update weights and biases */
void LayerFull::gpu_update(int realBatchSize, float lr)
{
    float step = -lr/realBatchSize;
#if CUDNN == 1
    CHECK_CUBLAS_ERRORS(cublasSaxpy(cublas_handle(), weights_->total_size(), &step, grad_weights_->getGpuPtr(), 1, weights_->getGpuPtr(), 1));
    CHECK_CUBLAS_ERRORS(cublasSaxpy(cublas_handle(), biases_->total_size(), &step, grad_biases_->getGpuPtr(), 1, biases_->getGpuPtr(), 1));
#else
    axpy_gpu(weights_, grad_weights_, step);
    axpy_gpu(biases_, grad_biases_, step);
#endif
}
#endif

/* initialize weights */
void LayerFull::initWeights(float *weights, int &offset)
{
    if (weights == NULL)
    {
        int fan_in = prev_layer_->sample_size_;
        float bound_weights = sqrt(6.0 / fan_in);
        float bound_biases = sqrt(1.0 / fan_in);

        default_random_engine generator;
        uniform_real_distribution<float> distribution_weights(bound_weights*-1, bound_weights);
        uniform_real_distribution<float> distribution_biases(bound_biases*-1, bound_biases);

        for (int i = 0; i < weights_->total_size(); i++)
        {
            weights_->data(i) = distribution_weights(generator);
        }

        for (int i = 0; i < biases_->total_size(); i++)
        {
            biases_->data(i) = distribution_biases(generator);
        }
    }
    else
    {
        for (int i = 0; i < weights_->total_size(); i++)
        {
            weights_->data(i) = weights[offset+i];
        }
        offset += weights_->total_size();

        for (int i = 0; i < biases_->total_size(); i++)
        {
            biases_->data(i) = weights[offset+i];
        }
        offset += biases_->total_size();
    }

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

    for (int i = 0; i < weights_->total_size(); i++)
    {
        weights[offset+i] = weights_->data(i);
    }
    offset += weights_->total_size();

    for (int i = 0; i < biases_->total_size(); i++)
    {
        weights[offset+i] = biases_->data(i);
    }
    offset += biases_->total_size();
}

/* get number of weights in this layer */
vector<int> LayerFull::getNumWeights()
{
    int nw = weights_->total_size();
    int nb = biases_->total_size();
    vector<int> num_weights;
    num_weights.push_back(nw+nb);
    num_weights.push_back(nw);
    num_weights.push_back(nb);
    return num_weights;
}
