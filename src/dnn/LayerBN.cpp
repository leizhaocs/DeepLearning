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
LayerBN::LayerBN(Params *params, Layer *prev_layer)
{
    prev_layer_ = prev_layer;

    type_ = "batchnorm";

    n_ = prev_layer_->n_;
    c_ = prev_layer_->c_;
    h_ = prev_layer_->h_;
    w_ = prev_layer_->w_;
    sample_size_ = c_ * h_ * w_;

    forwardTensor_ = new Tensor<float>(n_, c_, h_, w_);
    backwardTensor_ = new Tensor<float>(n_, c_, h_, w_);

    beta_ = new Tensor<float>(1, c_, 1, 1);
    gamma_ = new Tensor<float>(1, c_, 1, 1);

    grad_beta_ = new Tensor<float>(1, c_, 1, 1);
    grad_gamma_ = new Tensor<float>(1, c_, 1, 1);

    mean_ = new Tensor<float>(1, c_, 1, 1);
    var_ = new Tensor<float>(1, c_, 1, 1);

    running_mean_ = new Tensor<float>(1, c_, 1, 1);
    running_var_ = new Tensor<float>(1, c_, 1, 1);

    xhat_ = new Tensor<float>(n_, c_, h_, w_);
    grad_xhat_ = new Tensor<float>(n_, c_, h_, w_);

    temp1_ = new Tensor<float>(1, c_, 1, 1);
    temp2_ = new Tensor<float>(1, c_, 1, 1);
}

/* destructor */
LayerBN::~LayerBN()
{
    delete forwardTensor_;
    delete backwardTensor_;
    delete beta_;
    delete gamma_;
    delete grad_beta_;
    delete grad_gamma_;
    delete mean_;
    delete var_;
    delete running_mean_;
    delete running_var_;
    delete xhat_;
    delete grad_xhat_;
    delete temp1_;
    delete temp2_;
}

/* forward propagation */
void LayerBN::cpu_forward(int realBatchSize, bool train)
{
    clear(forwardTensor_);

    if (train)
    {
        mean_keep_channel(mean_, prev_layer_->forwardTensor_);
        variance_keep_channel(var_, prev_layer_->forwardTensor_, mean_);
        normalize_expand_channel(xhat_, prev_layer_->forwardTensor_, mean_, var_);
        scale_shift_expand_channel(forwardTensor_, xhat_, gamma_, beta_);

        add_with_momentum(running_mean_, mean_, 0.99);
        add_with_momentum(running_var_, var_, 0.99);
    }
    else
    {
        normalize_expand_channel(xhat_, prev_layer_->forwardTensor_, running_mean_, running_var_);
        scale_shift_expand_channel(forwardTensor_, xhat_, gamma_, beta_);
    }
}

/* backward propagation */
void LayerBN::cpu_backward(int realBatchSize)
{
    clear(prev_layer_->backwardTensor_);
    clear(grad_beta_);
    clear(grad_gamma_);

    mult_expand_channel(grad_xhat_, backwardTensor_, gamma_);
    backward_batchnorm(prev_layer_->backwardTensor_, grad_xhat_, xhat_, var_, temp1_, temp2_);

    sum_keep_channel(grad_beta_, backwardTensor_);
    product_sum_keep_channel(grad_gamma_, backwardTensor_, xhat_);
}

/* update weights and biases */
void LayerBN::cpu_update(int realBatchSize, float lr)
{
    float step = -lr/realBatchSize;
    axpy(beta_, grad_beta_, step);
    axpy(gamma_, grad_gamma_, step);
}

#if GPU == 1
/* forward propagation */
void LayerBN::gpu_forward(int realBatchSize, bool train)
{
    clear_gpu(forwardTensor_);

    if (train)
    {
        mean_keep_channel_gpu(mean_, prev_layer_->forwardTensor_);
        variance_keep_channel_gpu(var_, prev_layer_->forwardTensor_, mean_);
        normalize_expand_channel_gpu(xhat_, prev_layer_->forwardTensor_, mean_, var_);
        scale_shift_expand_channel_gpu(forwardTensor_, xhat_, gamma_, beta_);

        add_with_momentum_gpu(running_mean_, mean_, 0.99);
        add_with_momentum_gpu(running_var_, var_, 0.99);
    }
    else
    {
        normalize_expand_channel_gpu(xhat_, prev_layer_->forwardTensor_, running_mean_, running_var_);
        scale_shift_expand_channel_gpu(forwardTensor_, xhat_, gamma_, beta_);
    }
}

/* backward propagation */
void LayerBN::gpu_backward(int realBatchSize)
{
    clear_gpu(prev_layer_->backwardTensor_);
    clear_gpu(grad_beta_);
    clear_gpu(grad_gamma_);

    mult_expand_channel_gpu(grad_xhat_, backwardTensor_, gamma_);
    backward_batchnorm_gpu(prev_layer_->backwardTensor_, grad_xhat_, xhat_, var_, temp1_, temp2_);

    sum_keep_channel_gpu(grad_beta_, backwardTensor_);
    product_sum_keep_channel_gpu(grad_gamma_, backwardTensor_, xhat_);
}

/* update weights and biases */
void LayerBN::gpu_update(int realBatchSize, float lr)
{
    float step = -lr/realBatchSize;
    axpy_gpu(beta_, grad_beta_, step);
    axpy_gpu(gamma_, grad_gamma_, step);
}
#endif

/* initialize weights */
void LayerBN::initWeights(float *weights, int &offset)
{
    if (weights == NULL)
    {
        for (int i = 0; i < beta_->total_size(); i++)
        {
            beta_->data(i) = 0;
        }

        for (int i = 0; i < gamma_->total_size(); i++)
        {
            gamma_->data(i) = 1;
        }

        for (int i = 0; i < running_mean_->total_size(); i++)
        {
            running_mean_->data(i) = 0;
        }

        for (int i = 0; i < running_var_->total_size(); i++)
        {
            running_var_->data(i) = 1;
        }
    }
    else
    {
        for (int i = 0; i < beta_->total_size(); i++)
        {
            beta_->data(i) = weights[offset+i];
        }
        offset += beta_->total_size();

        for (int i = 0; i < gamma_->total_size(); i++)
        {
            gamma_->data(i) = weights[offset+i];
        }
        offset += gamma_->total_size();

        for (int i = 0; i < running_mean_->total_size(); i++)
        {
            running_mean_->data(i) = weights[offset+i];
        }
        offset += running_mean_->total_size();

        for (int i = 0; i < running_var_->total_size(); i++)
        {
            running_var_->data(i) = weights[offset+i];
        }
        offset += running_var_->total_size();
    }

#if GPU == 1
    if (use_gpu)
    {
        beta_->toGpu();
        gamma_->toGpu();
        running_mean_->toGpu();
        running_var_->toGpu();
    }
#endif
}

/* get weights */
void LayerBN::getWeights(float *weights, int &offset)
{
#if GPU == 1
    if (use_gpu)
    {
        beta_->toCpu();
        gamma_->toCpu();
        running_mean_->toCpu();
        running_var_->toCpu();
    }
#endif

    for (int i = 0; i < beta_->total_size(); i++)
    {
       weights[offset+i] = beta_->data(i);
    }
    offset += beta_->total_size();

    for (int i = 0; i < gamma_->total_size(); i++)
    {
       weights[offset+i] = gamma_->data(i);
    }
    offset += gamma_->total_size();

    for (int i = 0; i < running_mean_->total_size(); i++)
    {
       weights[offset+i] = running_mean_->data(i);
    }
    offset += running_mean_->total_size();

    for (int i = 0; i < running_var_->total_size(); i++)
    {
       weights[offset+i] = running_var_->data(i);
    }
    offset += running_var_->total_size();
}

/* get number of weights in this layer */
std::vector<int> LayerBN::getNumWeights()
{
    int nb = beta_->total_size();
    int ng = gamma_->total_size();
    int nm = running_mean_->total_size();
    int nv = running_var_->total_size();
    vector<int> num_weights;
    num_weights.push_back(nb+ng+nm+nv);
    num_weights.push_back(nb);
    num_weights.push_back(ng);
    num_weights.push_back(nm);
    num_weights.push_back(nv);
    return num_weights;
}
