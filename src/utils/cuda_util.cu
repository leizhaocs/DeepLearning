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

__global__ void dropout_kernel(float *input, float *output, int N, float *rand, float rate)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id < N)
    {
        output[id] = (rand[id] < rate) ? 0 : input[id];
    }
}

/* dropout */
void dropout_gpu(float *input, float *output, int n, int batch, float *rand, float rate)
{
    int N = n * batch;

    dropout_kernel<<<cuda_gridsize(N), BLOCK>>>(input, output, N, rand, rate);
    check_cuda_error();
}

__global__ void backward_dropout_kernel(float *backward_input, float *backward_output, float *rand, float rate, int N)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id < N)
    {
        backward_output[id] = (rand[id] < rate) ? 0 : backward_input[id];
    }
}

/* backward of dropout */
void backward_dropout_gpu(float *backward_input, float *backward_output, float *rand, float rate, int n, int batch)
{
    int N = n * batch;

    backward_dropout_kernel<<<cuda_gridsize(N), BLOCK>>>(backward_input, backward_output, rand, rate, N);
    check_cuda_error();
}

__global__ void maxpool_kernel(int N, int in_h, int in_w, int in_c, int out_h, int out_w,
    int stride_h, int stride_w, int filter_h, int filter_w, int padding_h, int padding_w,
    float *input, float *output, int *index)
{
    int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= N)
    {
        return;
    }

    int w = id % out_w;
    id /= out_w;
    int h = id % out_h;
    id /= out_h;
    int c = id % in_c;
    id /= in_c;
    int n = id;

    float max = -INFINITY;
    int max_i = -1;
    for (int i = 0; i < filter_h; i++)
    {
        for (int j = 0; j < filter_w; j++)
        {
            int cur_h = h * stride_h + i - padding_h;
            int cur_w = w * stride_w + j - padding_w;

            int in_index = ((n*in_c + c)*in_h + cur_h)*in_w + cur_w;
            int valid = (cur_h >= 0 && cur_h < in_h && cur_w >= 0 && cur_w < in_w);
            float val = (valid != 0) ? input[in_index] : -INFINITY;
            max_i = (val > max) ? in_index : max_i;
            max = (val > max) ? val : max;
        }
    }

    int out_index = ((n*in_c + c)*out_h+ h)*out_w + w;
    output[out_index] = max;
    index[out_index] = max_i;
}

/* max pooling */
void maxpool_gpu(float *input, float *output, int *index, int in_h, int in_w, int in_c, int out_h, int out_w,
    int stride_h, int stride_w, int filter_h, int filter_w, int padding_h, int padding_w, int batch)
{
    int N = out_h * out_w * in_c * batch;

    maxpool_kernel<<<cuda_gridsize(N), BLOCK>>>(N, in_h, in_w, in_c, out_h, out_w,
        stride_h, stride_w, filter_h, filter_w, padding_h, padding_w,
        input, output, index);
    check_cuda_error();
}

__global__ void backward_maxpool_kernel(int N, int in_h, int in_w, int in_c, int out_h, int out_w,
    int stride_h, int stride_w, int filter_h, int filter_w, int padding_h, int padding_w,
    float *backward_input, float *backward_output, int *index)
{
    int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= N)
    {
        return;
    }

    int in_index = id;
    int w = id % in_w;
    id /= in_w;
    int h = id % in_h;
    id /= in_h;
    int c = id % in_c;
    id /= in_c;
    int n = id;

    float d = 0;
    int area_h = (filter_h - 1) / stride_h;
    int area_w = (filter_w - 1) / stride_w;
    for (int l = -area_h; l < area_h + 1; l++)
    {
        for (int m = -area_w; m < area_w + 1; m++)
        {
            int cur_h = (h + padding_h) / stride_h + l;
            int cur_w = (w + padding_w) / stride_w + m;

            int out_index = ((n*in_c + c)*out_h + cur_h)*out_w + cur_w;
            int valid = (cur_w >= 0 && cur_w < out_w && cur_h >= 0 && cur_h < out_h);
            if (valid && index[out_index] == in_index)
            {
                d += backward_input[out_index];
            }
        }
    }
    backward_output[in_index] += d;
}

/* backward of maxpool */
void backward_maxpool_gpu(float *backward_input, float *backward_output, int *index, int in_h, int in_w, int in_c, int out_h, int out_w,
    int stride_h, int stride_w, int filter_h, int filter_w, int padding_h, int padding_w, int batch)
{
    int N = in_c * in_h * in_w * batch;

    backward_maxpool_kernel<<<cuda_gridsize(N), BLOCK>>>(N, in_h, in_w, in_c, out_h, out_w,
        stride_h, stride_w, filter_h, filter_w, padding_h, padding_w,
        backward_input, backward_output, index);
    check_cuda_error();
}

__global__ void relu_kernel(float *input, float *output, int n)
{
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n)
    {
        output[i] = (input[i] > 0) ? input[i] : 0;
    }
}

/* relu */
void relu_gpu(float *input, float *output, int n)
{
    relu_kernel<<<cuda_gridsize(n), BLOCK>>>(input, output, n);
    check_cuda_error();
}

__global__ void backward_relu_kernel(float *backward_input, float *forward_output, float *backward_output, int n)
{
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n)
    {
        backward_output[i] = (forward_output[i] > 0) ? backward_input[i] : 0;
    }
}

/* backward of relu */
void backward_relu_gpu(float *backward_input, float *forward_output, float *backward_output, int n)
{
    backward_relu_kernel<<<cuda_gridsize(n), BLOCK>>>(backward_input, forward_output, backward_output, n);
    check_cuda_error();
}

__global__ void sigmoid_kernel(float *input, float *output, int n)
{
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n)
    {
        float temp = expf((float)input[i]);
        output[i] = temp / (temp + 1);
    }
}

/* sigmoid */
void sigmoid_gpu(float *input, float *output, int n)
{
    sigmoid_kernel<<<cuda_gridsize(n), BLOCK>>>(input, output, n);
    check_cuda_error();
}

__global__ void backward_sigmoid_kernel(float *backward_input, float *forward_output, float *backward_output, int n)
{
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n)
    {
        backward_output[i] = backward_input[i] * forward_output[i] * (1 - forward_output[i]);
    }
}

/* backward of sigmoid */
void backward_sigmoid_gpu(float *backward_input, float *forward_output, float *backward_output, int n)
{
    backward_sigmoid_kernel<<<cuda_gridsize(n), BLOCK>>>(backward_input, forward_output, backward_output, n);
    check_cuda_error();
}

__global__ void softmax_kernel(float *input, float *output, int n, int batch)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= batch)
    {
        return;
    }

    float *in = input + id*n;
    float *out = output + id*n;

    float largest = -INFINITY;
    for (int i = 0; i < n; i++)
    {
        float val = in[i];
        largest = (val>largest) ? val : largest;
    }
    float sum = 0;
    for (int i = 0; i < n; i++)
    {
        float e = expf(in[i] - largest);
        sum += e;
        out[i] = e;
    }
    for (int i = 0; i < n; i++)
    {
        out[i] /= sum;
    }
}

/* softmax */
void softmax_gpu(float *input, float *output, int n, int batch)
{
    softmax_kernel<<<cuda_gridsize(batch), BLOCK>>>(input, output, n, batch);
    check_cuda_error();
}

__global__ void backward_softmax_kernel(float *backward_input, float *forward_output, float *backward_output, int n, int batch)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= batch)
    {
        return;
    }

    float *backward_in = backward_input + id*n;
    float *forward_out = forward_output + id*n;
    float *backward_out = backward_output + id*n;

    for (int i = 0; i < n; i++)
    {
        backward_out[i] = forward_out[i] - (backward_in[i] != 0);
    }
}

/* backward of softmax */
void backward_softmax_gpu(float *backward_input, float *forward_output, float *backward_output, int n, int batch)
{
    backward_softmax_kernel<<<cuda_gridsize(batch), BLOCK>>>(backward_input, forward_output, backward_output, n, batch);
    check_cuda_error();
}

/* grid size */
dim3 cuda_gridsize(int n)
{
    int k = (n-1) / BLOCK + 1;
    int x = k;
    int y = 1;
    if (x > 65535)
    {
        x = ceil(sqrt(k));
        y = (n-1)/(x*BLOCK) + 1;
    }
    dim3 d = {(unsigned int)x, (unsigned int)y, 1};
    return d;
}

/* check cuda error */
void check_cuda_error()
{                     
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("Cuda failure %s:%d:\n",__FILE__,__LINE__);
        exit(1);
    }
}

/* check cublas error */
void check_cublas_error(cublasStatus_t status)
{                     
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        printf("Cuda failure %s:%d\n",__FILE__,__LINE__);
        exit(1);
    }
}

/* get blas handle */
cublasHandle_t blas_handle()
{
    static int init = 0;
    static cublasHandle_t handle;
    if (!init)
    {
        cublasCreate(&handle);
        init = 1;
    }
    return handle;
}
