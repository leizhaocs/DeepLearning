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

__global__ void clear_kernel(float *X, int total_size)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= total_size)
    {
        return;
    }

    X[index] = 0;
}

/* X = 0 */
void clear_gpu(Tensor<float> *X)
{
    float *X_ptr = X->getGpuPtr();
    int total_size = X->total_size();

    clear_kernel<<<GRID(total_size), BLOCK>>>(X_ptr, total_size);
    CHECK_CUDA_ERRORS();
}

__global__ void random_kernel(float *X, int total_size)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= total_size)
    {
        return;
    }

    curandState state;
    curand_init(clock64(), index, 0, &state);
    X[index] = curand_uniform(&state);
}

/* X = rand(0, 1) */
void random_gpu(Tensor<float> *X)
{
    float *X_ptr = X->getGpuPtr();
    int total_size = X->total_size();

    random_kernel<<<GRID(total_size), BLOCK>>>(X_ptr, total_size);
    CHECK_CUDA_ERRORS();
}

__global__ void add_expand_channel_kernel(float *Y, float *X, int batch_size, int channels, int plane_size)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= channels*plane_size*batch_size)
    {
        return;
    }

    int c = (index / plane_size) % channels;

    Y[index] += X[c];
}

/* Y += X, expand X's channel dimension */
void add_expand_channel_gpu(Tensor<float> *Y, Tensor<float> *X)
{
    float *Y_ptr = Y->getGpuPtr();
    float *X_ptr = X->getGpuPtr();
    int batch_size = Y->getN();
    int channels = Y->getC();
    int plane_size = Y->plane_size();
    int num = channels*plane_size*batch_size;

    add_expand_channel_kernel<<<GRID(num), BLOCK>>>(Y_ptr, X_ptr, batch_size, channels, plane_size);
    CHECK_CUDA_ERRORS();
}

__global__ void axpy_kernel(float *Y, float *X, float ALPHA, int total_size)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= total_size)
    {
        return;
    }

    Y[index] += ALPHA*X[index];
}

/* Y += ALPHA * X */
void axpy_gpu(Tensor<float> *Y, Tensor<float> *X, float ALPHA)
{
    float *Y_ptr = Y->getGpuPtr();
    float *X_ptr = X->getGpuPtr();
    int total_size = Y->total_size();

    axpy_kernel<<<GRID(total_size), BLOCK>>>(Y_ptr, X_ptr, ALPHA, total_size);
    CHECK_CUDA_ERRORS();
}

/* Y = X */
void assign_gpu(Tensor<float> *Y, Tensor<float> *X)
{
    float *Y_ptr = Y->getGpuPtr();
    float *X_ptr = X->getGpuPtr();
    int total_size = Y->total_size();

    cudaMemcpy(Y_ptr, X_ptr, total_size*sizeof(float), cudaMemcpyDeviceToDevice);
}

__global__ void assign_cond_kernel(float *Y, float *X, float *R, float S, int total_size)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= total_size)
    {
        return;
    }

    Y[index] = (R[index] < S) ? 0 : X[index];
}

/* Y = (R < S) ? 0 : X */
void assign_cond_gpu(Tensor<float> *Y, Tensor<float> *X, Tensor<float> *R, float S)
{
    float *Y_ptr = Y->getGpuPtr();
    float *X_ptr = X->getGpuPtr();
    float *R_ptr = R->getGpuPtr();
    int total_size = Y->total_size();

    assign_cond_kernel<<<GRID(total_size), BLOCK>>>(Y_ptr, X_ptr, R_ptr, S, total_size);
    CHECK_CUDA_ERRORS();
}

__global__ void mean_keep_channel_kernel(float *Y, float *X, int batch_size, int channels, int plane_size)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= channels)
    {
        return;
    }

    int N = batch_size * plane_size;

    Y[index] = 0;
    for (int b = 0; b < batch_size; b++)
    {
        for (int i = 0; i < plane_size; i++)
        {
            int xindex = (b*channels + index)*plane_size + i;
            Y[index] += X[xindex];
        }
    }
    Y[index] /= N;
}

/* Y = mean(X), preserve the channel dimension */
void mean_keep_channel_gpu(Tensor<float> *Y, Tensor<float> *X)
{
    float *Y_ptr = Y->getGpuPtr();
    float *X_ptr = X->getGpuPtr();
    int batch_size = X->getN();
    int channels = X->getC();
    int plane_size = X->plane_size();

    mean_keep_channel_kernel<<<GRID(channels), BLOCK>>>(Y_ptr, X_ptr, batch_size, channels, plane_size);
    CHECK_CUDA_ERRORS();
}

__global__ void variance_keep_channel_kernel(float *Y, float *X, float *MEAN, int batch_size, int channels, int plane_size)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= channels)
    {
        return;
    }

    int N = batch_size * plane_size;

    Y[index] = 0;
    for (int b = 0; b < batch_size; b++)
    {
        for (int i = 0; i < plane_size; i++)
        {
            int xindex = (b*channels + index)*plane_size + i;
            float temp = X[xindex] - MEAN[index];
            Y[index] += temp * temp;
        }
    }
    Y[index] /= N;
}

/* Y = variance(X, MEAN), preserve the channel dimension */
void variance_keep_channel_gpu(Tensor<float> *Y, Tensor<float> *X, Tensor<float> *MEAN)
{
    float *Y_ptr = Y->getGpuPtr();
    float *X_ptr = X->getGpuPtr();
    float *MEAN_ptr = MEAN->getGpuPtr();
    int batch_size = X->getN();
    int channels = X->getC();
    int plane_size = X->plane_size();

    variance_keep_channel_kernel<<<GRID(channels), BLOCK>>>(Y_ptr, X_ptr, MEAN_ptr, batch_size, channels, plane_size);
    CHECK_CUDA_ERRORS();
}

__global__ void normalize_expand_channel_kernel(float *Y, float *X, float *MEAN, float *VAR, int batch_size, int channels, int plane_size)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= channels*plane_size*batch_size)
    {
        return;
    }

    int c = (index / plane_size) % channels;

    Y[index] = (X[index] - MEAN[c]) / sqrt(VAR[c] + EPSILON);
}

/* Y = normalize(X, MEAN, VAR), expand MEAN's and VAR's channel dimension */
void normalize_expand_channel_gpu(Tensor<float> *Y, Tensor<float> *X, Tensor<float> *MEAN, Tensor<float> *VAR)
{
    float *Y_ptr = Y->getGpuPtr();
    float *X_ptr = X->getGpuPtr();
    float *MEAN_ptr = MEAN->getGpuPtr();
    float *VAR_ptr = VAR->getGpuPtr();
    int batch_size = Y->getN();
    int channels = Y->getC();
    int plane_size = Y->plane_size();
    int num = channels*plane_size*batch_size;

    normalize_expand_channel_kernel<<<GRID(num), BLOCK>>>(Y_ptr, X_ptr, MEAN_ptr, VAR_ptr, batch_size, channels, plane_size);
    CHECK_CUDA_ERRORS();
}

__global__ void scale_shift_expand_channel_kernel(float *Y, float *X, float *GAMMA, float *BETA, int batch_size, int channels, int plane_size)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= channels*plane_size*batch_size)
    {
        return;
    }

    int c = (index / plane_size) % channels;

    Y[index] = X[index] * GAMMA[c] + BETA[c];
}


/* Y = GAMMA * X + BETA, expand GAMMA's and BETA's channel dimension */
void scale_shift_expand_channel_gpu(Tensor<float> *Y, Tensor<float> *X, Tensor<float> *GAMMA, Tensor<float> *BETA)
{
    float *Y_ptr = Y->getGpuPtr();
    float *X_ptr = X->getGpuPtr();
    float *GAMMA_ptr = GAMMA->getGpuPtr();
    float *BETA_ptr = BETA->getGpuPtr();
    int batch_size = Y->getN();
    int channels = Y->getC();
    int plane_size = Y->plane_size();
    int num = channels*plane_size*batch_size;

    scale_shift_expand_channel_kernel<<<GRID(num), BLOCK>>>(Y_ptr, X_ptr, GAMMA_ptr, BETA_ptr, batch_size, channels, plane_size);
    CHECK_CUDA_ERRORS();
}

__global__ void add_with_momentum_kernel(float *Y, float *X, float M, int total_size)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= total_size)
    {
        return;
    }

    Y[index] = Y[index] * M + X[index] * (1 - M);
}

/* Y = Y * M + X * (1 - M) */
void add_with_momentum_gpu(Tensor<float> *Y, Tensor<float> *X, float M)
{
    float *Y_ptr = Y->getGpuPtr();
    float *X_ptr = X->getGpuPtr();
    int total_size = Y->total_size();

    add_with_momentum_kernel<<<GRID(total_size), BLOCK>>>(Y_ptr, X_ptr, M, total_size);
    CHECK_CUDA_ERRORS();
}

__global__ void mult_expand_channel_kernel(float *Y, float *X1, float *X2, int batch_size, int channels, int plane_size)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= channels*plane_size*batch_size)
    {
        return;
    }

    int c = (index / plane_size) % channels;

    Y[index] = X1[index] * X2[c];
}

/* Y = X1 * X2, expand X2's channel dimension */
void mult_expand_channel_gpu(Tensor<float> *Y, Tensor<float> *X1, Tensor<float> *X2)
{
    float *Y_ptr = Y->getGpuPtr();
    float *X1_ptr = X1->getGpuPtr();
    float *X2_ptr = X2->getGpuPtr();
    int batch_size = Y->getN();
    int channels = Y->getC();
    int plane_size = Y->plane_size();
    int num = channels*plane_size*batch_size;

    mult_expand_channel_kernel<<<GRID(num), BLOCK>>>(Y_ptr, X1_ptr, X2_ptr, batch_size, channels, plane_size);
    CHECK_CUDA_ERRORS();
}

__global__ void sum_keep_channel_kernel(float *Y, float *X, int batch_size, int channels, int plane_size)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= channels)
    {
        return;
    }

    Y[index] = 0;
    for (int b = 0; b < batch_size; b++)
    {
        for (int i = 0; i < plane_size; i++)
        {
            int xindex = (b*channels + index)*plane_size + i;
            Y[index] += X[xindex];
        }
    }
}

/* Y = sum(X), preserve the channel dimension */
void sum_keep_channel_gpu(Tensor<float> *Y, Tensor<float> *X)
{
    float *Y_ptr = Y->getGpuPtr();
    float *X_ptr = X->getGpuPtr();
    int batch_size = X->getN();
    int channels = X->getC();
    int plane_size = X->plane_size();

    sum_keep_channel_kernel<<<GRID(channels), BLOCK>>>(Y_ptr, X_ptr, batch_size, channels, plane_size);
    CHECK_CUDA_ERRORS();
}

__global__ void product_sum_keep_channel_kernel(float *Y, float *X1, float *X2, int batch_size, int channels, int plane_size)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= channels)
    {
        return;
    }

    Y[index] = 0;
    for (int b = 0; b < batch_size; b++)
    {
        for (int i = 0; i < plane_size; i++)
        {
            int xindex = (b*channels + index)*plane_size + i;
            Y[index] += X1[xindex] * X2[xindex];
        }
    }
}

/* Y = sum(X1 * X2), preserve the channel dimension */
void product_sum_keep_channel_gpu(Tensor<float> *Y, Tensor<float> *X1, Tensor<float> *X2)
{
    float *Y_ptr = Y->getGpuPtr();
    float *X1_ptr = X1->getGpuPtr();
    float *X2_ptr = X2->getGpuPtr();
    int batch_size = X1->getN();
    int channels = X1->getC();
    int plane_size = X1->plane_size();

    product_sum_keep_channel_kernel<<<GRID(channels), BLOCK>>>(Y_ptr, X1_ptr, X2_ptr, batch_size, channels, plane_size);
    CHECK_CUDA_ERRORS();
}

__global__ void backward_batchnorm_temp_kernel(float *T1, float *T2, float *DXHAT, float *XHAT, int batch_size, int channels, int plane_size)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= channels)
    {
        return;
    }

    T1[index] = 0;
    T2[index] = 0;
    for (int b = 0; b < batch_size; b++)
    {
        for (int i = 0; i < plane_size; i++)
        {
            int xindex = (b*channels + index)*plane_size + i;
            T1[index] += DXHAT[xindex];
            T2[index] += DXHAT[xindex] * XHAT[xindex];
        }
    }
}

__global__ void backward_batchnorm_kernel(float *DX, float *DXHAT, float *XHAT, float *VAR, float *T1, float *T2, int batch_size, int channels, int plane_size)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= channels*plane_size*batch_size)
    {
        return;
    }

    int N = batch_size * plane_size;
    int c = (index / plane_size) % channels;

    DX[index] = (1.0/(N*sqrt(VAR[c]+EPSILON))) * (DXHAT[index]*N - T1[c] - T1[c]*T2[c]*XHAT[index]);
}

/* backward of batch normalization */
void backward_batchnorm_gpu(Tensor<float> *DX, Tensor<float> *DXHAT, Tensor<float> *XHAT, Tensor<float> *VAR, Tensor<float> *T1, Tensor<float> *T2)
{
    float *DX_ptr = DX->getGpuPtr();
    float *DXHAT_ptr = DXHAT->getGpuPtr();
    float *XHAT_ptr = XHAT->getGpuPtr();
    float *VAR_ptr = VAR->getGpuPtr();
    float *T1_ptr = T1->getGpuPtr();
    float *T2_ptr = T2->getGpuPtr();
    int batch_size = DX->getN();
    int channels = DX->getC();
    int plane_size = DX->plane_size();
    int num = channels*plane_size*batch_size;

    backward_batchnorm_temp_kernel<<<GRID(channels), BLOCK>>>(T1_ptr, T2_ptr, DXHAT_ptr, XHAT_ptr, batch_size, channels, plane_size);
    backward_batchnorm_kernel<<<GRID(num), BLOCK>>>(DX_ptr, DXHAT_ptr, XHAT_ptr, VAR_ptr, T1_ptr, T2_ptr, batch_size, channels, plane_size);
    CHECK_CUDA_ERRORS();
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
    dqn_target_kernel<<<GRID(batch_size), BLOCK>>>(targets, outputs, lambda, rewards, actions, final_states, batch_size, n);
    CHECK_CUDA_ERRORS();
}
