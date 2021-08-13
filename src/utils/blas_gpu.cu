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

__global__ void add_bias_kernel(float *output, float *biases, int batch, int n, int size)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= n*size*batch)
    {
        return;
    }

    int i = index % size;
    index /= size;
    int j = index % n;
    index /= n;
    int k = index;

    output[(k*n+j)*size + i] += biases[j];
}

/* add biases in gpu */
void add_bias_gpu(float *output, float *biases, int batch, int n, int size)
{
    int num = n*size*batch;

    add_bias_kernel<<<cuda_gridsize(num), BLOCK>>>(output, biases, batch, n, size);
    check_cuda_error();
}

__global__ void backward_bias_conn_kernel(float *grad_biases, float *delta, int batch, int n)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= n)
    {
        return;
    }

    float sum = 0;
    for (int b = 0; b < batch; b++)
    {
        int i = b*n + index;
        sum += delta[i];
    }
    grad_biases[index] += sum;
}

__global__ void backward_bias_kernel(float *grad_biases, float *delta, int batch, int n, int size)
{
    __shared__ float part[BLOCK];

    int filter = blockIdx.x;
    int p = threadIdx.x;
    float sum = 0;
    for (int b = 0; b < batch; b++)
    {
        for (int i = 0; i < size; i += BLOCK)
        {
            int index = p + i + size*(filter + n*b);
            sum += (p+i < size) ? delta[index] : 0;
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

/* calculate gradient of biases in gpu */
void backward_bias_gpu(float *grad_biases, float *delta, int batch, int n, int size)
{
    if(size == 1)
    {
        backward_bias_conn_kernel<<<cuda_gridsize(n), BLOCK>>>(grad_biases, delta, batch, n);
    }
    else
    {
        backward_bias_kernel<<<n, BLOCK>>>(grad_biases, delta, batch, n, size);
    }
    check_cuda_error();
}

__global__ void axpy_kernel(int N, float ALPHA, float *X,  float *Y)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i < N)
    {
        Y[i] += ALPHA*X[i];
    }
}

/* Y += ALPHA * X */
void axpy_gpu(int N, float ALPHA, float *X, float *Y)
{
    axpy_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, Y);
    check_cuda_error();
}

__global__ void clear_kernel(int N, float *X)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i < N)
    {
        X[i] = 0;
    }
}

/* clear all elements to 0 */
void clear_gpu(int N, float *X)
{
    clear_kernel<<<cuda_gridsize(N), BLOCK>>>(N, X);
    check_cuda_error();
}

__global__ void random_kernel(int N, float *X)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i < N)
    {
        curandState state;
        curand_init(clock64(), i, 0, &state);
        X[i] = curand_uniform(&state);
    }
}

/* set all elements to random number between [0, 1] */
void random_gpu(int N, float *X)
{
    random_kernel<<<cuda_gridsize(N), BLOCK>>>(N, X);
    check_cuda_error();
}

/* copy */
void copy_gpu(float *dest, float *src, int number_of_float)
{
    cudaMemcpy(dest, src, number_of_float*sizeof(float), cudaMemcpyDeviceToDevice);
}

__global__ void dqn_target_kernel(float *targets, float *outputs, float lambda, float *rewards, int *actions, int *final_states, int batch_size, int n)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= batch_size)
    {
        return;
    }

    float *t = targets + id*n;
    float *o = outputs + id*n;
    float r = rewards[id];
    int a = actions[id];
    int f = final_states[id];

    float largest = -INFINITY;
    for (int i = 0; i < n; i++)
    {
        float val = t[i];
        largest = (val>largest) ? val : largest;
    }

    for (int i = 0; i < n; i++)
    {
        t[i] = o[i];
    }

    if (f)
    {
        t[a] = 0;
    }
    else
    {
        t[a] = largest*lambda + r;
    }
}

/* calculate TD target for dqn */
void dqn_target_gpu(float *targets, float *outputs, float lambda, float *rewards, int *actions, int *final_states, int batch_size, int n)
{
    dqn_target_kernel<<<cuda_gridsize(batch_size), BLOCK>>>(targets, outputs, lambda, rewards, actions, final_states, batch_size, n);
    check_cuda_error();
}
