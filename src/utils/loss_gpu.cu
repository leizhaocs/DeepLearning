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

__global__ void cross_entropy_kernel(float *errors, float *outputs, float *targets, int n)
{
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n)
    {
        errors[i] = log(outputs[i]+EPSILON) * targets[i] * -1;
    }
}

/* cross entropy loss on gpu */
void cross_entropy_gpu(Tensor<float> *errors, Tensor<float> *outputs, Tensor<float> *targets, int targets_offset, int batch_size, int classes)
{
    int n = batch_size * classes;
    cross_entropy_kernel<<<cuda_gridsize(n), BLOCK>>>(errors->getGpuPtr(), outputs->getGpuPtr(), targets->getGpuPtr()+targets_offset*classes, n);
    check_cuda_error();
}

__global__ void mse_kernel(float *errors, float *outputs, float *targets, int n)
{
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n)
    {
        errors[i] = outputs[i] - targets[i];
    }
}

/* mean squared loss on gpu */
void mse_gpu(Tensor<float> *errors, Tensor<float> *outputs, Tensor<float> *targets, int targets_offset, int batch_size, int classes)
{
    int n = batch_size * classes;
    mse_kernel<<<cuda_gridsize(n), BLOCK>>>(errors->getGpuPtr(), outputs->getGpuPtr(), targets->getGpuPtr()+targets_offset*classes, n);
    check_cuda_error();
}
