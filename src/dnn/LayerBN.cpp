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
{/*
    prev_layer_ = prev_layer;

    type_ = "batchnormalization";

    Assert(params->hasField("channels"), "Batch Normalization Layer must have channels specified.");
    channels_ = params->getScalari("channels");

    shape_ = prev_layer->shape_;

    forwardTensor_ = new Tensor(shape_);
    backwardTensor_ = new Tensor(shape_);

    std::vector<int> bn_shape = std::vector<int> {channels_};
    mean_ = new Tensor(bn_shape);
    std_ = new Tensor(bn_shape);
    beta_ = new Tensor(bn_shape);
    gamma_ = new Tensor(bn_shape);
    running_mean_ = new Tensor(bn_shape);
    running_std_ = new Tensor(bn_shape);
    grad_beta_ = new Tensor(bn_shape);
    grad_gamma_ = new Tensor(bn_shape);*/
}

/* destructor */
LayerBN::~LayerBN()
{/*
    delete forwardTensor_;
    delete backwardTensor_;
    delete mean_;
    delete std_;
    delete beta_;
    delete gamma_;
    delete running_mean_;
    delete running_std_;
    delete grad_beta_;
    delete grad_gamma_;*/
}

/* forward propagation */
void LayerBN::cpu_forward(int realBatchSize, bool train)
{/*
    if (!train)
    {
        if (shape_.size() == 2)
        {
            #pragma omp parallel for num_threads(OPENMP_THREADS)
            for (int n = 0; n < realBatchSize; n++)
            {
                for (int l = 0; l < shape_[1]; l++)
                {
                    forwardTensor_->data(n, l) = prev_layer_->forwardTensor_->data(n, l);
                    forwardTensor_->data(n, l) -= running_mean_->data(l);
                    forwardTensor_->data(n, l) /= running_std_->data(l);
                    forwardTensor_->data(n, l) *= gamma_->data(l);
                    forwardTensor_->data(n, l) += beta_->data(l);
                }
            }
        }
        else if (shape_.size() == 4)
        {
            #pragma omp parallel for num_threads(OPENMP_THREADS)
            for (int n = 0; n < realBatchSize; n++)
            {
                for (int h = 0; h < shape_[1]; h++)
                {
                    for (int w = 0; w < shape_[2]; w++)
                    {
                        for (int c = 0; c < shape_[3]; c++)
                        {
                            forwardTensor_->data(n, h, w, c) = prev_layer_->forwardTensor_->data(n, h, w, c);
                            forwardTensor_->data(n, h, w, c) -= running_mean_->data(c);
                            forwardTensor_->data(n, h, w, c) /= running_std_->data(c);
                            forwardTensor_->data(n, h, w, c) *= gamma_->data(c);
                            forwardTensor_->data(n, h, w, c) += beta_->data(c);
                        }
                    }
                }
            }
        }
        else
        {
            Assert(false, "Wrong size in Batch Normalization.");
        }
    }
    else
    {
        if (shape_.size() == 2)
        {
            #pragma omp parallel for num_threads(OPENMP_THREADS)
            for (int l = 0; l < shape_[1]; l++)
            {
                mean_->data(l) = 0;
                for (int n = 0; n < realBatchSize; n++)
                {
                    mean_->data(l) += prev_layer_->forwardTensor_->data(n, l);
                }
                mean_->data(l) /= realBatchSize;

                std_->data(l) = 0;
                for (int n = 0; n < realBatchSize; n++)
                {
                    std_->data(l) += (prev_layer_->forwardTensor_->data(n, l)-mean_->data(l)) * (prev_layer_->forwardTensor_->data(n, l)-mean_->data(l));
                }
                std_->data(l) /= realBatchSize;

                for (int n = 0; n < realBatchSize; n++)
                {
                    forwardTensor_->data(n, l) = prev_layer_->forwardTensor_->data(n, l) - mean_->data(l);
                    forwardTensor_->data(n, l) /= sqrt((double)(std_->data(l)*std_->data(l) + EPSILON));
                    forwardTensor_->data(n, l) = forwardTensor_->data(n, l)*gamma_->data(l) + beta_->data(l);
                }
            }
        }
        else if (shape_.size() == 4)
        {
            #pragma omp parallel for num_threads(OPENMP_THREADS)
            for (int c = 0; c < shape_[3]; c++)
            {
                mean_->data(c) = 0;
                for (int h = 0; h < shape_[1]; h++)
                {
                    for (int w = 0; w < shape_[2]; w++)
                    {
                        for (int n = 0; n < realBatchSize; n++)
                        {
                            mean_->data(c) += prev_layer_->forwardTensor_->data(n, h, w, c);
                        }
                    }
                }
                mean_->data(c) /= realBatchSize * shape_[1] * shape_[2];

                std_->data(c) = 0;
                for (int h = 0; h < shape_[1]; h++)
                {
                    for (int w = 0; w < shape_[2]; w++)
                    {
                        for (int n = 0; n < realBatchSize; n++)
                        {
                            std_->data(c) += (prev_layer_->forwardTensor_->data(n, h, w, c)-mean_->data(c)) * (prev_layer_->forwardTensor_->data(n, h, w, c)-mean_->data(c));
                        }
                    }
                }
                std_->data(c) /= realBatchSize * shape_[1] * shape_[2];

                for (int h = 0; h < shape_[1]; h++)
                {
                    for (int w = 0; w < shape_[2]; w++)
                    {
                        for (int n = 0; n < realBatchSize; n++)
                        {
                            forwardTensor_->data(n, h, w, c) = prev_layer_->forwardTensor_->data(n, h, w, c) - mean_->data(c);
                            forwardTensor_->data(n, h, w, c) /= sqrt((double)(std_->data(c)*std_->data(c) + EPSILON));
                            forwardTensor_->data(n, h, w, c) = forwardTensor_->data(n, h, w, c)*gamma_->data(c) + beta_->data(c);
                        }
                    }
                }
            }
        }
        else
        {
            Assert(false, "Wrong size in Batch Normalization.");
        }
    }*/
}

/* backward propagation */
void LayerBN::cpu_backward(int realBatchSize)
{/*
    if (prev_layer_->type_ == "input")
    {
        return;
    }

    if (shape_.size() == 2)
    {
        #pragma omp parallel for num_threads(OPENMP_THREADS)
        for (int n = 0; n < realBatchSize; n++)
        {
            for (int l = 0; l < shape_[1]; l++)
            {
                prev_layer_->backwardTensor_->data(n, l) = 0;
                for (int i = 0; i < realBatchSize; i++)
                {
                    prev_layer_->backwardTensor_->data(n, l) += backwardTensor_->data(i, l) * (forwardTensor_->data(i, l) - mean_->data(l));
                }
                prev_layer_->backwardTensor_->data(n, l) *= (mean_->data(l) - forwardTensor_->data(n, l)) / (std_->data(l) + EPSILON);
                for (int i = 0; i < realBatchSize; i++)
                {
                    prev_layer_->backwardTensor_->data(n, l) -= backwardTensor_->data(i, l);
                }
                prev_layer_->backwardTensor_->data(n, l) += realBatchSize * backwardTensor_->data(n, l);
                prev_layer_->backwardTensor_->data(n, l) *= gamma_->data(l) / sqrt((double)(std_->data(l)+EPSILON)) / realBatchSize;
            }
        }
    }
    else if (shape_.size() == 4)
    {
        #pragma omp parallel for num_threads(OPENMP_THREADS)
        for (int n = 0; n < realBatchSize; n++)
        {
            for (int h = 0; h < shape_[1]; h++)
            {
                for (int w = 0; w < shape_[2]; w++)
                {
                    for (int c = 0; c < shape_[3]; c++)
                    {
                        prev_layer_->backwardTensor_->data(n, h, w, c) = 0;
                        for (int i = 0; i < realBatchSize; i++)
                        {
                            for (int j = 0; j < shape_[1]; j++)
                            {
                                for (int k = 0; k < shape_[2]; k++)
                                {
                                    prev_layer_->backwardTensor_->data(n, h, w, c) += backwardTensor_->data(i, j, k, c) * (forwardTensor_->data(i, j, k, c) - mean_->data(c));
                                }
                            }
                        }
                        prev_layer_->backwardTensor_->data(n, h, w, c) *= (mean_->data(c) - forwardTensor_->data(n, h, w, c)) / (std_->data(c) + EPSILON);
                        for (int i = 0; i < realBatchSize; i++)
                        {
                            for (int j = 0; j < shape_[1]; j++)
                            {
                                for (int k = 0; k < shape_[2]; k++)
                                {
                                    prev_layer_->backwardTensor_->data(n, h, w, c) -= backwardTensor_->data(i, j, k, c);
                                }
                            }
                        }
                        prev_layer_->backwardTensor_->data(n, h, w, c) += realBatchSize * shape_[1] * shape_[2] * backwardTensor_->data(n, h, w, c);
                        prev_layer_->backwardTensor_->data(n, h, w, c) *= gamma_->data(c) / sqrt((double)(std_->data(c)+EPSILON)) / (realBatchSize * shape_[1] * shape_[2]);
                    }
                }
            }
        }
    }
    else
    {
        Assert(false, "Wrong size in Batch Normalization.");
    }*/
}

/* calculate gradients */
//void LayerBN::cpu_calGrads(int realBatchSize)
//{
/*
    if (shape_.size() == 2)
    {
        #pragma omp parallel for num_threads(OPENMP_THREADS)
        for (int l = 0; l < shape_[1]; l++)
        {
            grad_beta_->data(l) = 0;
            for (int n = 0; n < realBatchSize; n++)
            {
                grad_beta_->data(l) += backwardTensor_->data(n, l);
            }
            grad_beta_->data(l) /= realBatchSize;
        }
        #pragma omp parallel for num_threads(OPENMP_THREADS)
        for (int l = 0; l < shape_[1]; l++)
        {
            grad_gamma_->data(l) = 0;
            for (int n = 0; n < realBatchSize; n++)
            {
                grad_gamma_->data(l) += backwardTensor_->data(n, l) * (forwardTensor_->data(n, l)-mean_->data(l)) / sqrt((double)(std_->data(l)+EPSILON));
            }
            grad_gamma_->data(l) /= realBatchSize;
        }
    }
    else if (shape_.size() == 4)
    {
        #pragma omp parallel for num_threads(OPENMP_THREADS)
        for (int c = 0; c < shape_[3]; c++)
        {
            grad_beta_->data(c) = 0;
            for (int i = 0; i < realBatchSize; i++)
            {
                for (int j = 0; j < shape_[1]; j++)
                {
                    for (int k = 0; k < shape_[2]; k++)
                    {
                        grad_beta_->data(c) += backwardTensor_->data(i, j, k, c);
                    }
                }
            }
            grad_beta_->data(c) /= realBatchSize * shape_[1] * shape_[2];
        }
        #pragma omp parallel for num_threads(OPENMP_THREADS)
        for (int c = 0; c < shape_[3]; c++)
        {
            grad_gamma_->data(c) = 0;
            for (int i = 0; i < realBatchSize; i++)
            {
                for (int j = 0; j < shape_[1]; j++)
                {
                    for (int k = 0; k < shape_[2]; k++)
                    {
                        grad_gamma_->data(c) += backwardTensor_->data(i, j, k, c) * (forwardTensor_->data(i, j, k, c)-mean_->data(c)) / sqrt((double)(std_->data(c)+EPSILON));
                    }
                }
            }
            grad_gamma_->data(c) /= realBatchSize * shape_[1] * shape_[2];
        }
    }
    else
    {
        Assert(false, "Wrong size in Batch Normalization.");
    }*/
//}

/* update weights and biases */
void LayerBN::cpu_update(int realBatchSize, float lr)
{/*
    if (shape_.size() == 2)
    {
        #pragma omp parallel for num_threads(OPENMP_THREADS)
        for (int l = 0; l < shape_[1]; l++)
        {
            beta_->data(l) -= lr * grad_beta_->data(l);
        }
        #pragma omp parallel for num_threads(OPENMP_THREADS)
        for (int l = 0; l < shape_[1]; l++)
        {
            gamma_->data(l) -= lr * grad_gamma_->data(l);
        }
        #pragma omp parallel for num_threads(OPENMP_THREADS)
        for (int l = 0; l < shape_[1]; l++)
        {
            running_mean_->data(l) = 0.9*running_mean_->data(l) + 0.1*mean_->data(l);
        }
        #pragma omp parallel for num_threads(OPENMP_THREADS)
        for (int l = 0; l < shape_[1]; l++)
        {
            running_std_->data(l) = 0.9*running_std_->data(l) + 0.1*std_->data(l);
        }
    }
    else if (shape_.size() == 4)
    {
        #pragma omp parallel for num_threads(OPENMP_THREADS)
        for (int c = 0; c < shape_[3]; c++)
        {
            beta_->data(c) -= lr * grad_beta_->data(c);
        }
        #pragma omp parallel for num_threads(OPENMP_THREADS)
        for (int c = 0; c < shape_[3]; c++)
        {
            gamma_->data(c) -= lr * grad_gamma_->data(c);
        }
        #pragma omp parallel for num_threads(OPENMP_THREADS)
        for (int c = 0; c < shape_[3]; c++)
        {
            running_mean_->data(c) = 0.9*running_mean_->data(c) + 0.1*mean_->data(c);
        }
        #pragma omp parallel for num_threads(OPENMP_THREADS)
        for (int c = 0; c < shape_[3]; c++)
        {
            running_std_->data(c) = 0.9*running_std_->data(c) + 0.1*std_->data(c);
        }
    }
    else
    {
        Assert(false, "Wrong size in Batch Normalization.");
    }*/
}

#if GPU == 1
/* forward propagation */
void LayerBN::gpu_forward(int realBatchSize, bool train)
{
}

/* backward propagation */
void LayerBN::gpu_backward(int realBatchSize)
{
}

/* update weights and biases */
void LayerBN::gpu_update(int realBatchSize, float lr)
{
}
#endif

/* initialize weights */
void LayerBN::initWeights(float *weights, int &offset)
{/*
    running_mean_->copyIn(weights+offset, running_mean_->size());
    offset += running_mean_->size();

    running_std_->copyIn(weights+offset, running_std_->size());
    offset += running_std_->size();

    beta_->copyIn(weights+offset, beta_->size());
    offset += beta_->size();

    gamma_->copyIn(weights+offset, gamma_->size());
    offset += gamma_->size();*/
}

/* get weights */
void LayerBN::getWeights(float *weights, int &offset)
{/*
    running_mean_->copyOut(weights+offset, running_mean_->size());
    offset += running_mean_->size();

    running_std_->copyOut(weights+offset, running_std_->size());
    offset += running_std_->size();

    beta_->copyOut(weights+offset, beta_->size());
    offset += beta_->size();

    gamma_->copyOut(weights+offset, gamma_->size());
    offset += gamma_->size();*/
}

/* get number of weights in this layer */
std::vector<int> LayerBN::getNumWeights()
{/*
    int total = running_mean_->size() + running_std_->size() + beta_->size() + gamma_->size();
    std::vector<int> num_weights;
    num_weights.push_back(total);
    num_weights.push_back(running_mean_->size());
    num_weights.push_back(running_std_->size());
    num_weights.push_back(beta_->size());
    num_weights.push_back(gamma_->size());
    return num_weights;*/
    std::vector<int> num_weights;
    return num_weights;
}
