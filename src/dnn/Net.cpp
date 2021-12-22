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
Net::Net(Params *layer_params, int layers_num, int batchSize, string loss_func)
{
    num_layers_ = layers_num;
    layers_.resize(num_layers_);
    loss_ = loss_func;

    Params *layer_param = &(layer_params[0]);
    string layer_type = layer_param->getString("type");
    Assert(layer_type == "input", "The first layer must be input layer");
    layers_[0] = new LayerInput(layer_param, batchSize);

    for (int i = 1; i < num_layers_; i++)
    {
        Layer *prev_layer = layers_[i-1];
        layer_param = &(layer_params[i]);
        layer_type = layer_param->getString("type");
        if (layer_type == "convolution")
        {
            layers_[i] = new LayerConv(layer_param, prev_layer);
        }
        else if (layer_type == "pool")
        {
            layers_[i] = new LayerPool(layer_param, prev_layer);
        }
        else if (layer_type == "full")
        {
            layers_[i] = new LayerFull(layer_param, prev_layer);
        }
        else if (layer_type == "activation")
        {
            layers_[i] = new LayerAct(layer_param, prev_layer);
        }
        else if (layer_type == "batchnorm")
        {
            layers_[i] = new LayerBN(layer_param, prev_layer);
        }
        else if (layer_type == "dropout")
        {
            layers_[i] = new LayerDrop(layer_param, prev_layer);
        }
        else
        {
            Assert(false, layer_type + " - unknown type of the layer");
        }
        printf("%15s (%d  %d  %d  %d)\n", layer_type.c_str(), layers_[i]->n_, layers_[i]->c_, layers_[i]->h_, layers_[i]->w_);
    }
}

/* destructor */
Net::~Net()
{
    for (int i = 0; i < num_layers_; ++i)
    {
        delete layers_[i];
    }
    layers_.clear();
}

/* initialize weights for all layers */
void Net::initWeights(float *weights)
{
    int offset = 0;
    for (int i = 0; i < layers_.size(); i++)
    {
        layers_[i]->initWeights(weights, offset);
    }
}

/* get weights from all layers */
void Net::getWeights(float *weights)
{
    int offset = 0;
    for (int i = 0; i < layers_.size(); i++)
    {
        layers_[i]->getWeights(weights, offset);
    }
}

/* get number of weights per layer, only return layers with weights, 0: real 1; binary */
int Net::getNumWeights(vector<vector<int>> *total_weights)
{
    int total = 0;

    for (int i = 0; i < layers_.size(); i++)
    {
        vector<int> num_weights = layers_[i]->getNumWeights();

        total += num_weights[0];

        if (total_weights != NULL)
        {
            vector<int> temp;
            for (int j = 0; j < num_weights.size(); j++)
            {
                temp.push_back(num_weights[j]);
            }
            (*total_weights).push_back(temp);
        }
    }

    return total;
}

/* init the input of the first layer, (channels, height, width) match the dimension of first layer */
void Net::initForward(Tensor<float> *inputs, int inputs_offset, int forwardTensor_offset, int realBatchSize, int channels, int height, int width)
{
#if GPU == 1
    if (use_gpu)
    {
        int image_size = channels * height * width;
        float *inputs_start = inputs->getGpuPtr() + inputs_offset*image_size;
        float *forwardTensor_start = layers_[0]->forwardTensor_->getGpuPtr() + forwardTensor_offset*image_size;
        cudaMemcpy(forwardTensor_start, inputs_start, realBatchSize*image_size*sizeof(float), cudaMemcpyDeviceToDevice);
    }
    else
    {
        int image_size = channels * height * width;
        float *inputs_start = inputs->getCpuPtr() + inputs_offset*image_size;
        float *forwardTensor_start = layers_[0]->forwardTensor_->getCpuPtr() + forwardTensor_offset*image_size;
        memcpy(forwardTensor_start, inputs_start, realBatchSize*image_size*sizeof(float));
    }
#else
    int image_size = channels * height * width;
    float *inputs_start = inputs->getCpuPtr() + inputs_offset*image_size;
    float *forwardTensor_start = layers_[0]->forwardTensor_->getCpuPtr() + forwardTensor_offset*image_size;
    memcpy(forwardTensor_start, inputs_start, realBatchSize*image_size*sizeof(float));
#endif
}

/* forward propagate through all layers */
void Net::forward(int realBatchSize, bool train)
{
    for (int i = 0; i < layers_.size(); i++)
    {
#if GPU == 1
        if (use_gpu)
        {
            layers_[i]->gpu_forward(realBatchSize, train);
        }
        else
        {
            layers_[i]->cpu_forward(realBatchSize, train);
        }
#else
        layers_[i]->cpu_forward(realBatchSize, train);
#endif
    }
}

/* calculate the loss before backward propagation */
void Net::loss(Tensor<float> *targets, int targets_offset, int realBatchSize, int classes)
{
#if GPU == 1
    if (use_gpu)
    {
        if (loss_.compare("cross_entropy") == 0)
        {
            cross_entropy_gpu(layers_.back()->backwardTensor_, layers_.back()->forwardTensor_, targets, targets_offset);
        }
        else if (loss_.compare("mse") == 0)
        {
            mse_gpu(layers_.back()->backwardTensor_, layers_.back()->forwardTensor_, targets, targets_offset);
        }
        else
        {
            Assert(false, "Unsupported loss function");
        }
    }
    else
    {
        if (loss_.compare("cross_entropy") == 0)
        {
            cross_entropy(layers_.back()->backwardTensor_, layers_.back()->forwardTensor_, targets, targets_offset);
        }
        else if (loss_.compare("mse") == 0)
        {
            mse(layers_.back()->backwardTensor_, layers_.back()->forwardTensor_, targets, targets_offset);
        }
        else
        {
            Assert(false, "Unsupported loss function");
        }
    }
#else
    if (loss_.compare("cross_entropy") == 0)
    {
        cross_entropy(layers_.back()->backwardTensor_, layers_.back()->forwardTensor_, targets, targets_offset);
    }
    else if (loss_.compare("mse") == 0)
    {
        mse(layers_.back()->backwardTensor_, layers_.back()->forwardTensor_, targets, targets_offset);
    }
    else
    {
        Assert(false, "Unsupported loss function");
    }
#endif
}

/* init the backward input of the last layer, i.e. calculate the derivative of loss function */
void Net::initBackward(Tensor<float> *targets, int targets_offset, int realBatchSize, int classes)
{
#if GPU == 1
    if (use_gpu)
    {
        if (loss_.compare("cross_entropy") == 0)
        {
            backward_cross_entropy_gpu(layers_.back()->backwardTensor_, layers_.back()->forwardTensor_, targets, targets_offset);
        }
        else if (loss_.compare("mse") == 0)
        {
            backward_mse_gpu(layers_.back()->backwardTensor_, layers_.back()->forwardTensor_, targets, targets_offset);
        }
        else
        {
            Assert(false, "Unsupported loss function");
        }
    }
    else
    {
        if (loss_.compare("cross_entropy") == 0)
        {
            backward_cross_entropy(layers_.back()->backwardTensor_, layers_.back()->forwardTensor_, targets, targets_offset);
        }
        else if (loss_.compare("mse") == 0)
        {
            backward_mse(layers_.back()->backwardTensor_, layers_.back()->forwardTensor_, targets, targets_offset);
        }
        else
        {
            Assert(false, "Unsupported loss function");
        }
    }
#else
    if (loss_.compare("cross_entropy") == 0)
    {
        backward_cross_entropy(layers_.back()->backwardTensor_, layers_.back()->forwardTensor_, targets, targets_offset);
    }
    else if (loss_.compare("mse") == 0)
    {
        backward_mse(layers_.back()->backwardTensor_, layers_.back()->forwardTensor_, targets, targets_offset);
    }
    else
    {
        Assert(false, "Unsupported loss function");
    }
#endif
}

/* backward propagate through all layers */
void Net::backward(int realBatchSize)
{
    for (int i = layers_.size()-1; i >= 0; i--)
    {
#if GPU == 1
        if (use_gpu)
        {
            layers_[i]->gpu_backward(realBatchSize);
        }
        else
        {
            layers_[i]->cpu_backward(realBatchSize);
        }
#else
        layers_[i]->cpu_backward(realBatchSize);
#endif
    }
}

/* update weights and biases */
void Net::update(int realBatchSize, float lr)
{
    for (int i = layers_.size()-1; i >= 0; i--)
    {
#if GPU == 1
        if (use_gpu)
        {
            layers_[i]->gpu_update(realBatchSize, lr);
        }
        else
        {
            layers_[i]->cpu_update(realBatchSize, lr);
        }
#else
        layers_[i]->cpu_update(realBatchSize, lr);
#endif
    }
}

/* get the outputs of the final layer, classes must match last layer */
void Net::getOutputs(Tensor<float> *outputs, int outputs_offset, int forwardTensor_offset, int realBatchSize, int classes)
{
#if GPU == 1
    if (use_gpu)
    {
        float *outputs_start = outputs->getGpuPtr() + outputs_offset*classes;
        float *forwardTensor_start = layers_.back()->forwardTensor_->getGpuPtr() + forwardTensor_offset*classes;
        cudaMemcpy(outputs_start, forwardTensor_start, realBatchSize*classes*sizeof(float), cudaMemcpyDeviceToDevice);
    }
    else
    {
        float *outputs_start = outputs->getCpuPtr() + outputs_offset*classes;
        float *forwardTensor_start = layers_.back()->forwardTensor_->getCpuPtr() + forwardTensor_offset*classes;
        memcpy(outputs_start, forwardTensor_start, realBatchSize*classes*sizeof(float));
    }
#else
    float *outputs_start = outputs->getCpuPtr() + outputs_offset*classes;
    float *forwardTensor_start = layers_.back()->forwardTensor_->getCpuPtr() + forwardTensor_offset*classes;
    memcpy(outputs_start, forwardTensor_start, realBatchSize*classes*sizeof(float));
#endif
}
