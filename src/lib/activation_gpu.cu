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

__global__ void relu_kernel(float *input, float *output, int total_size)
{
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < total_size)
    {
        output[i] = (input[i] > 0) ? input[i] : 0;
    }
}

/* relu */
void relu_gpu(Tensor<float> *input, Tensor<float> *output)
{
    float *input_ptr = input->getGpuPtr();
    float *output_ptr = output->getGpuPtr();
    int total_size = output->total_size();

    relu_kernel<<<GRID(total_size), BLOCK>>>(input_ptr, output_ptr, total_size);
    CHECK_CUDA_ERRORS();
}

__global__ void backward_relu_kernel(float *backward_input, float *forward_output, float *backward_output, int total_size)
{
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < total_size)
    {
        backward_output[i] = (forward_output[i] > 0) ? backward_input[i] : 0;
    }
}

/* backward of relu */
void backward_relu_gpu(Tensor<float> *backward_input, Tensor<float> *forward_output, Tensor<float> *backward_output)
{
    float *backward_input_ptr = backward_input->getGpuPtr();
    float *forward_output_ptr = forward_output->getGpuPtr();
    float *backward_output_ptr = backward_output->getGpuPtr();
    int total_size = backward_output->total_size();

    backward_relu_kernel<<<GRID(total_size), BLOCK>>>(backward_input_ptr, forward_output_ptr, backward_output_ptr, total_size);
    CHECK_CUDA_ERRORS();
}

__global__ void sigmoid_kernel(float *input, float *output, int total_size)
{
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < total_size)
    {
        float temp = expf((float)input[i]);
        output[i] = temp / (temp + 1);
    }
}

/* sigmoid */
void sigmoid_gpu(Tensor<float> *input, Tensor<float> *output)
{
    float *input_ptr = input->getGpuPtr();
    float *output_ptr = output->getGpuPtr();
    int total_size = output->total_size();

    sigmoid_kernel<<<GRID(total_size), BLOCK>>>(input_ptr, output_ptr, total_size);
    CHECK_CUDA_ERRORS();
}

__global__ void backward_sigmoid_kernel(float *backward_input, float *forward_output, float *backward_output, int total_size)
{
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < total_size)
    {
        backward_output[i] = backward_input[i] * forward_output[i] * (1 - forward_output[i]);
    }
}

/* backward of sigmoid */
void backward_sigmoid_gpu(Tensor<float> *backward_input, Tensor<float> *forward_output, Tensor<float> *backward_output)
{
    float *backward_input_ptr = backward_input->getGpuPtr();
    float *forward_output_ptr = forward_output->getGpuPtr();
    float *backward_output_ptr = backward_output->getGpuPtr();
    int total_size = backward_output->total_size();

    backward_sigmoid_kernel<<<GRID(total_size), BLOCK>>>(backward_input_ptr, forward_output_ptr, backward_output_ptr, total_size);
    CHECK_CUDA_ERRORS();
}

__global__ void tanh_kernel(float *input, float *output, int total_size)
{
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < total_size)
    {
        float temp = expf((float)input[i]);
        float temp_inverse = 1 / temp;
        output[i] = (temp - temp_inverse) / (temp + temp_inverse);
    }
}

/* tanh */
void tanh_gpu(Tensor<float> *input, Tensor<float> *output)
{
    float *input_ptr = input->getGpuPtr();
    float *output_ptr = output->getGpuPtr();
    int total_size = output->total_size();

    tanh_kernel<<<GRID(total_size), BLOCK>>>(input_ptr, output_ptr, total_size);
    CHECK_CUDA_ERRORS();
}

__global__ void backward_tanh_kernel(float *backward_input, float *forward_output, float *backward_output, int total_size)
{
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < total_size)
    {
        backward_output[i] = backward_input[i] * (1 - forward_output[i] * forward_output[i]);
    }
}

/* backward of tanh */
void backward_tanh_gpu(Tensor<float> *backward_input, Tensor<float> *forward_output, Tensor<float> *backward_output)
{
    float *backward_input_ptr = backward_input->getGpuPtr();
    float *forward_output_ptr = forward_output->getGpuPtr();
    float *backward_output_ptr = backward_output->getGpuPtr();
    int total_size = backward_output->total_size();

    backward_tanh_kernel<<<GRID(total_size), BLOCK>>>(backward_input_ptr, forward_output_ptr, backward_output_ptr, total_size);
    CHECK_CUDA_ERRORS();
}

__global__ void softmax_kernel(float *input, float *output, int sample_size, int batch_size)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= batch_size)
    {
        return;
    }

    float *in = input + id*sample_size;
    float *out = output + id*sample_size;

    float largest = -INFINITY;
    for (int i = 0; i < sample_size; i++)
    {
        float val = in[i];
        largest = (val>largest) ? val : largest;
    }
    float sum = 0;
    for (int i = 0; i < sample_size; i++)
    {
        float e = expf(in[i] - largest);
        sum += e;
        out[i] = e;
    }
    for (int i = 0; i < sample_size; i++)
    {
        out[i] /= sum;
    }
}

/* softmax */
void softmax_gpu(Tensor<float> *input, Tensor<float> *output)
{
    float *input_ptr = input->getGpuPtr();
    float *output_ptr = output->getGpuPtr();
    int batch_size = output->getN();
    int sample_size = output->sample_size();

    softmax_kernel<<<GRID(batch_size), BLOCK>>>(input_ptr, output_ptr, sample_size, batch_size);
    CHECK_CUDA_ERRORS();
}

/*FIXME: implement the optimized derivative computation for softmax+cross_entropy
__global__ void backward_softmax_kernel1(float *backward_input, float *forward_output, float *backward_output, int total_size)
{
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < total_size)
    {
        backward_output[i] = forward_output[i] - (backward_input[i] != 0);
    }
}
*/

__global__ void backward_softmax_kernel(float *backward_input, float *forward_output, float *backward_output, int sample_size, int batch_size)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= batch_size)
    {
        return;
    }

    float *backward_in = backward_input + id*sample_size;
    float *forward_out = forward_output + id*sample_size;
    float *backward_out = backward_output + id*sample_size;

    for (int i = 0; i < sample_size; i++)
    {
        backward_out[i] = 0;
        for (int j = 0; j < sample_size; j++)
        {
            backward_out[i] += -1 * forward_out[i] * (forward_out[j] - (i==j)) * backward_in[j];
        }
    }
}

/* backward of softmax */
void backward_softmax_gpu(Tensor<float> *backward_input, Tensor<float> *forward_output, Tensor<float> *backward_output)
{
    float *backward_input_ptr = backward_input->getGpuPtr();
    float *forward_output_ptr = forward_output->getGpuPtr();
    float *backward_output_ptr = backward_output->getGpuPtr();
    int batch_size = backward_output->getN();
    int sample_size = backward_output->sample_size();

    backward_softmax_kernel<<<GRID(batch_size), BLOCK>>>(backward_input_ptr, forward_output_ptr, backward_output_ptr, sample_size, batch_size);
    CHECK_CUDA_ERRORS();
}
