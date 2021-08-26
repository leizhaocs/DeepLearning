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

__global__ void backward_bias_conn_kernel(float *grad_biases, float *delta, int batch_size, int channels)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= channels)
    {
        return;
    }

    float sum = 0;
    for (int b = 0; b < batch_size; b++)
    {
        int i = b*channels + index;
        sum += delta[i];
    }
    grad_biases[index] += sum;
}

__global__ void backward_bias_kernel(float *grad_biases, float *delta, int batch_size, int channels, int plane_size)
{
    __shared__ float part[BLOCK];

    int filter = blockIdx.x;
    int p = threadIdx.x;
    float sum = 0;
    for (int b = 0; b < batch_size; b++)
    {
        for (int i = 0; i < plane_size; i += BLOCK)
        {
            int index = p + i + plane_size*(filter + channels*b);
            sum += (p+i < plane_size) ? delta[index] : 0;
        }
    }
    part[p] = sum;
    __syncthreads();
    if (p == 0)
    {
        for(int i = 0; i < BLOCK; ++i)
        {
            grad_biases[filter] += part[i];
        }
    }
}

/* calculate gradient of biases */
void backward_bias_gpu(Tensor<float> *grad_biases, Tensor<float> *delta)
{
    float *grad_biases_ptr = grad_biases->getGpuPtr();
    float *delta_ptr = delta->getGpuPtr();
    int batch_size = delta->getN();
    int channels = delta->getC();
    int plane_size = delta->plane_size();

    if(plane_size == 1)
    {
        backward_bias_conn_kernel<<<GRID(channels), BLOCK>>>(grad_biases_ptr, delta_ptr, batch_size, channels);
    }
    else
    {
        backward_bias_kernel<<<GRID(channels), BLOCK>>>(grad_biases_ptr, delta_ptr, batch_size, channels, plane_size);
    }
    check_cuda_error();
}
