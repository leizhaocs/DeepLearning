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

__global__ void cross_entropy_kernel(float *errors, float *outputs, float *targets, int total_size)
{
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < total_size)
    {
        errors[i] = log(outputs[i]+EPSILON) * targets[i] * -1;
    }
}

/* FIXME: correctly calculate the loss */
/* cross entropy loss on gpu */
void cross_entropy_gpu(Tensor<float> *errors, Tensor<float> *outputs, Tensor<float> *targets, int targets_offset)
{
    int total_size = errors->total_size();
    int classes = errors->sample_size();
    float *errors_ptr = errors->getGpuPtr();
    float *outputs_ptr = outputs->getGpuPtr();
    float *targets_ptr = targets->getGpuPtr()+targets_offset*classes;

    cross_entropy_kernel<<<GRID(total_size), BLOCK>>>(errors_ptr, outputs_ptr, targets_ptr, total_size);
    CHECK_CUDA_ERRORS();
}

__global__ void backward_cross_entropy_kernel(float *errors, float *outputs, float *targets, int total_size)
{
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < total_size)
    {
        if (targets[i] != 0)
        {
            errors[i] = -1 / (outputs[i]+EPSILON);
        }
        else
        {
            errors[i] = 0;
        }
    }
}

/* backward of cross entropy loss on gpu */
void backward_cross_entropy_gpu(Tensor<float> *errors, Tensor<float> *outputs, Tensor<float> *targets, int targets_offset)
{
    int total_size = errors->total_size();
    int classes = errors->sample_size();
    float *errors_ptr = errors->getGpuPtr();
    float *outputs_ptr = outputs->getGpuPtr();
    float *targets_ptr = targets->getGpuPtr()+targets_offset*classes;

    backward_cross_entropy_kernel<<<GRID(total_size), BLOCK>>>(errors_ptr, outputs_ptr, targets_ptr, total_size);
    CHECK_CUDA_ERRORS();
}

__global__ void mse_kernel(float *errors, float *outputs, float *targets, int total_size)
{
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < total_size)
    {
        errors[i] = outputs[i] - targets[i];
    }
}

/* FIXME: correctly calculate the loss */
/* mean squared loss on gpu */
void mse_gpu(Tensor<float> *errors, Tensor<float> *outputs, Tensor<float> *targets, int targets_offset)
{
    int total_size = errors->total_size();
    int classes = errors->sample_size();
    float *errors_ptr = errors->getGpuPtr();
    float *outputs_ptr = outputs->getGpuPtr();
    float *targets_ptr = targets->getGpuPtr()+targets_offset*classes;

    mse_kernel<<<GRID(total_size), BLOCK>>>(errors_ptr, outputs_ptr, targets_ptr, total_size);
    CHECK_CUDA_ERRORS();
}

__global__ void backward_mse_kernel(float *errors, float *outputs, float *targets, int total_size)
{
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < total_size)
    {
        errors[i] = outputs[i] - targets[i];
    }
}

/* backward of mean squared loss on gpu */
void backward_mse_gpu(Tensor<float> *errors, Tensor<float> *outputs, Tensor<float> *targets, int targets_offset)
{
    int total_size = errors->total_size();
    int classes = errors->sample_size();
    float *errors_ptr = errors->getGpuPtr();
    float *outputs_ptr = outputs->getGpuPtr();
    float *targets_ptr = targets->getGpuPtr()+targets_offset*classes;

    backward_mse_kernel<<<GRID(total_size), BLOCK>>>(errors_ptr, outputs_ptr, targets_ptr, total_size);
    CHECK_CUDA_ERRORS();
}
