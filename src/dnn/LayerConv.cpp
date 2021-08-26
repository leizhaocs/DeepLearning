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
LayerConv::LayerConv(Params *params, Layer *prev_layer)
{
    prev_layer_ = prev_layer;

    type_ = "convolution";

    Assert(params->hasField("filterSize"), "Convolutional Layer must have filter size specified.");
    vector<int> filterSize = params->getVectori("filterSize");
    num_filters_= filterSize[0];
    filter_h_ = filterSize[1];
    filter_w_ = filterSize[2];

    Assert(params->hasField("stride"), "Convolutional Layer must have stride specified.");
    vector<int> stride = params->getVectori("stride");
    stride_h_ = stride[0];
    stride_w_ = stride[1];

    Assert(params->hasField("padding"), "Convolutional Layer must have padding specified.");
    vector<int> padding = params->getVectori("padding");
    padding_h_ = padding[0];
    padding_w_ = padding[1];

    n_ = prev_layer_->n_;
    c_ = num_filters_;
    h_ = (prev_layer_->h_-filter_h_+2*padding_h_)/stride_h_ + 1;
    w_ = (prev_layer_->w_-filter_w_+2*padding_w_)/stride_w_ + 1;
    sample_size_ = c_ * h_ * w_;

    forwardTensor_ = new Tensor<float>(n_, c_, h_, w_);
    backwardTensor_ = new Tensor<float>(n_, c_, h_, w_);

    filters_ = new Tensor<float>(num_filters_, prev_layer_->c_, filter_h_, filter_w_);
    biases_ = new Tensor<float>(1, 1, 1, num_filters_);

    grad_filters_ = new Tensor<float>(num_filters_, prev_layer_->c_, filter_h_, filter_w_);
    grad_biases_ = new Tensor<float>(1, 1, 1, num_filters_);
}

/* destructor */
LayerConv::~LayerConv()
{
    delete forwardTensor_;
    delete backwardTensor_;
    delete filters_;
    delete biases_;
    delete grad_filters_;
    delete grad_biases_;
}

/* forward propagation */
void LayerConv::cpu_forward(int realBatchSize, bool train)
{
    clear(forwardTensor_);

    int m = num_filters_;
    int n = h_ * w_;
    int k = filter_h_ * filter_w_ * prev_layer_->c_;

    Tensor<float> *atensor = new Tensor<float>(1, 1, k, n);
    for (int i = 0; i < realBatchSize; i++)
    {
        float *a = filters_->getCpuPtr();
        float *b = atensor->getCpuPtr();
        float *c = forwardTensor_->getCpuPtr() + i*sample_size_;

        float *im = prev_layer_->forwardTensor_->getCpuPtr() + i*prev_layer_->sample_size_;
        im2col_cpu(im, prev_layer_->c_, prev_layer_->h_, prev_layer_->w_, h_, w_,
                   filter_h_, filter_w_, stride_h_, stride_w_, padding_h_, padding_w_, b);
        gemm(0, 0, m, n, k, a, k, b, n, c, n);
    }
    delete atensor;
    add_expand_channel(forwardTensor_, biases_);
}

/* backward propagation */
void LayerConv::cpu_backward(int realBatchSize)
{
    clear(prev_layer_->backwardTensor_);
    clear(grad_filters_);
    clear(grad_biases_);

    int m = num_filters_;
    int n = filter_h_ * filter_w_ * prev_layer_->c_;
    int k = h_ * w_;

    Tensor<float> *atensor = new Tensor<float>(n, 1, 1, k);
    for (int i = 0; i < realBatchSize; i++)
    {
        clear(atensor);

        float *a = backwardTensor_->getCpuPtr() + i*sample_size_;
        float *b = atensor->getCpuPtr();
        float *c = grad_filters_->getCpuPtr();

        float *im = prev_layer_->forwardTensor_->getCpuPtr() + i*prev_layer_->sample_size_;
        im2col_cpu(im, prev_layer_->c_, prev_layer_->h_, prev_layer_->w_, h_, w_,
                   filter_h_, filter_w_, stride_h_, stride_w_, padding_h_, padding_w_, b);
        gemm(0, 1, m, n, k, a, k, b, k, c, n);

        clear(atensor);

        a = filters_->getCpuPtr();
        b = backwardTensor_->getCpuPtr() + i*sample_size_;
        c = atensor->getCpuPtr();

        gemm(1, 0, n, k, m, a, n, b, k, c, k);
        float *imd = prev_layer_->backwardTensor_->getCpuPtr() + i*prev_layer_->sample_size_;
        col2im_cpu(c, prev_layer_->c_, prev_layer_->h_, prev_layer_->w_, h_, w_,
                   filter_h_, filter_w_, stride_h_, stride_w_, padding_h_, padding_w_, imd);
    }
    delete atensor;
    backward_bias(grad_biases_, backwardTensor_);
}

/* update weights and biases */
void LayerConv::cpu_update(int realBatchSize, float lr)
{
    float step = -lr/realBatchSize;
    axpy(filters_, grad_filters_, step);
    axpy(biases_, grad_biases_, step);
}

#if GPU == 1
/* forward propagation */
void LayerConv::gpu_forward(int realBatchSize, bool train)
{
    clear_gpu(forwardTensor_);

    int m = num_filters_;
    int n = h_ * w_;
    int k = filter_h_ * filter_w_ * prev_layer_->c_;

    Tensor<float> *atensor = new Tensor<float>(n, 1, 1, k);
    for (int i = 0; i < realBatchSize; i++)
    {
        float *a = filters_->getGpuPtr();
        float *b = atensor->getGpuPtr();
        float *c = forwardTensor_->getGpuPtr() + i*sample_size_;

        float *im = prev_layer_->forwardTensor_->getGpuPtr() + i*prev_layer_->sample_size_;
        im2col_gpu(im, prev_layer_->c_, prev_layer_->h_, prev_layer_->w_, h_, w_,
                   filter_h_, filter_w_, stride_h_, stride_w_, padding_h_, padding_w_, b);
        gemm_gpu(0, 0, m, n, k, a, k, b, n, c, n);
    }
    delete atensor;
    add_expand_channel_gpu(forwardTensor_, biases_);
}

/* backward propagation */
void LayerConv::gpu_backward(int realBatchSize)
{
    clear_gpu(prev_layer_->backwardTensor_);
    clear_gpu(grad_filters_);
    clear_gpu(grad_biases_);

    int m = num_filters_;
    int n = filter_h_ * filter_w_ * prev_layer_->c_;
    int k = h_ * w_;

    Tensor<float> *atensor = new Tensor<float>(1, 1, k, n);
    for (int i = 0; i < realBatchSize; i++)
    {
        clear_gpu(atensor);

        float *a = backwardTensor_->getGpuPtr() + i*sample_size_;
        float *b = atensor->getGpuPtr();
        float *c = grad_filters_->getGpuPtr();

        float *im = prev_layer_->forwardTensor_->getGpuPtr() + i*prev_layer_->sample_size_;
        im2col_gpu(im, prev_layer_->c_, prev_layer_->h_, prev_layer_->w_, h_, w_,
                   filter_h_, filter_w_, stride_h_, stride_w_, padding_h_, padding_w_, b);
        gemm_gpu(0, 1, m, n, k, a, k, b, k, c, n);

        clear_gpu(atensor);

        a = filters_->getGpuPtr();
        b = backwardTensor_->getGpuPtr() + i*sample_size_;
        c = atensor->getGpuPtr();

        gemm_gpu(1, 0, n, k, m, a, n, b, k, c, k);
        float *imd = prev_layer_->backwardTensor_->getGpuPtr() + i*prev_layer_->sample_size_;
        col2im_gpu(c, prev_layer_->c_, prev_layer_->h_, prev_layer_->w_, h_, w_,
                   filter_h_, filter_w_, stride_h_, stride_w_, padding_h_, padding_w_, imd);
    }
    delete atensor;
    backward_bias_gpu(grad_biases_, backwardTensor_);
}

/* update weights and biases */
void LayerConv::gpu_update(int realBatchSize, float lr)
{
    float step = -lr/realBatchSize;

    axpy_gpu(filters_, grad_filters_, step);
    axpy_gpu(biases_, grad_biases_, step);
}
#endif

/* initialize weights */
void LayerConv::initWeights(float *weights, int &offset)
{
    if (weights == NULL)
    {
        int fan_in = prev_layer_->c_ * filter_h_ * filter_w_;
        float bound_filters = sqrt(6.0 / fan_in);
        float bound_biases = sqrt(1.0 / fan_in);

        default_random_engine generator;
        uniform_real_distribution<float> distribution_filters(bound_filters*-1, bound_filters);
        uniform_real_distribution<float> distribution_biases(bound_biases*-1, bound_biases);

        for (int i = 0; i < filters_->total_size(); i++)
        {
            filters_->data(i) = distribution_filters(generator);
        }

        for (int i = 0; i < biases_->total_size(); i++)
        {
            biases_->data(i) = distribution_biases(generator);
        }
    }
    else
    {
        for (int i = 0; i < filters_->total_size(); i++)
        {
            filters_->data(i) = weights[offset+i];
        }
        offset += filters_->total_size();

        for (int i = 0; i < biases_->total_size(); i++)
        {
            biases_->data(i) = weights[offset+i];
        }
        offset += biases_->total_size();
    }

#if GPU == 1
    if (use_gpu)
    {
        filters_->toGpu();
        biases_->toGpu();
    }
#endif
}

/* get weights */
void LayerConv::getWeights(float *weights, int &offset)
{
#if GPU == 1
    if (use_gpu)
    {
        filters_->toCpu();
        biases_->toCpu();
    }
#endif

    for (int i = 0; i < filters_->total_size(); i++)
    {
       weights[offset+i] = filters_->data(i);
    }
    offset += filters_->total_size();

    for (int i = 0; i < biases_->total_size(); i++)
    {
       weights[offset+i] = biases_->data(i);
    }
    offset += biases_->total_size();
}

/* get number of weights in this layer */
vector<int> LayerConv::getNumWeights()
{
    int nf = filters_->total_size();
    int nb = biases_->total_size();
    vector<int> num_weights;
    num_weights.push_back(nf+nb);
    num_weights.push_back(nf);
    num_weights.push_back(nb);
    return num_weights;
}
