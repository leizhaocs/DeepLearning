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

/* X = 0 */
void clear(Tensor<float> *X)
{
    int total_size = X->total_size();

    for (int i = 0; i < total_size; i++)
    {
        X->data(i) = 0;
    }
}

/* X = rand(0, 1) */
void random(Tensor<float> *X)
{
    int total_size = X->total_size();

    for (int i = 0; i < total_size; i++)
    {
        X->data(i) = rand()/float(RAND_MAX);
    }
}

/* Y += X, expand X's channel dimension */
void add_expand_channel(Tensor<float> *Y, Tensor<float> *X)
{
    int batch_size = Y->getN();
    int channels = Y->getC();
    int plane_size = Y->plane_size();

    for (int c = 0; c < channels; c++)
    {
        for (int b = 0; b < batch_size; b++)
        {
            for (int i = 0; i < plane_size; i++)
            {
                Y->data(b, c, i) += X->data(c);
            }
        }
    }
}

/* Y += ALPHA * X */
void axpy(Tensor<float> *Y, Tensor<float> *X, float ALPHA)
{
    int total_size = Y->total_size();

    for (int i = 0; i < total_size; i++)
    {
        Y->data(i) += ALPHA * X->data(i);
    }
}

/* Y = X */
void assign(Tensor<float> *Y, Tensor<float> *X)
{
    int total_size = Y->total_size();

    memcpy(Y->getCpuPtr(), X->getCpuPtr(), total_size*sizeof(float));
}

/* Y = (R < S) ? 0 : X */
void assign_cond(Tensor<float> *Y, Tensor<float> *X, Tensor<float> *R, float S)
{
    int total_size = Y->total_size();

    for (int i = 0; i < total_size; i++)
    {
        Y->data(i) = (R->data(i) < S) ? 0 : X->data(i);
    }
}

/* Y = mean(X), preserve the channel dimension */
void mean_keep_channel(Tensor<float> *Y, Tensor<float> *X)
{
    int batch_size = X->getN();
    int channels = X->getC();
    int plane_size = X->plane_size();

    int N = batch_size * plane_size;

    for (int c = 0; c < channels; c++)
    {
        Y->data(c) = 0;
        for (int b = 0; b < batch_size; b++)
        {
            for (int i = 0; i < plane_size; i++)
            {
                Y->data(c) += X->data(b, c, i);
            }
        }
        Y->data(c) /= N;
    }
}

/* Y = variance(X), preserve the channel dimension */
void variance_keep_channel(Tensor<float> *Y, Tensor<float> *X, Tensor<float> *MEAN)
{
    int batch_size = X->getN();
    int channels = X->getC();
    int plane_size = X->plane_size();

    int N = batch_size * plane_size;

    for (int c = 0; c < channels; c++)
    {
        Y->data(c) = 0;
        for (int b = 0; b < batch_size; b++)
        {
            for (int i = 0; i < plane_size; i++)
            {
                float temp = X->data(b, c, i) - MEAN->data(c);
                Y->data(c) += temp * temp;
            }
        }
        Y->data(c) /= N;
    }
}

/* Y = normalize(X, MEAN, VAR), expand MEAN's and VAR's channel dimension */
void normalize_expand_channel(Tensor<float> *Y, Tensor<float> *X, Tensor<float> *MEAN, Tensor<float> *VAR)
{
    int batch_size = Y->getN();
    int channels = Y->getC();
    int plane_size = Y->plane_size();

    for (int c = 0; c < channels; c++)
    {
        for (int b = 0; b < batch_size; b++)
        {
            for (int i = 0; i < plane_size; i++)
            {
                Y->data(b, c, i) = (X->data(b, c, i) - MEAN->data(c)) / sqrt(VAR->data(c) + EPSILON);
            }
        }
    }
}

/* Y = GAMMA * X + BETA, expand GAMMA's and BETA's channel dimension */
void scale_shift_expand_channel(Tensor<float> *Y, Tensor<float> *X, Tensor<float> *GAMMA, Tensor<float> *BETA)
{
    int batch_size = Y->getN();
    int channels = Y->getC();
    int plane_size = Y->plane_size();

    for (int c = 0; c < channels; c++)
    {
        for (int b = 0; b < batch_size; b++)
        {
            for (int i = 0; i < plane_size; i++)
            {
                Y->data(b, c, i) = X->data(b, c, i) * GAMMA->data(c) + BETA->data(c);
            }
        }
    }
}

/* Y = Y * M + X * (1 - M) */
void add_with_momentum(Tensor<float> *Y, Tensor<float> *X, float M)
{
    int total_size = Y->total_size();

    for (int i = 0; i < total_size; i++)
    {
        Y->data(i) = Y->data(i) * M + X->data(i) * (1 - M);
    }
}

/* Y = X1 * X2, expand X2's channel dimension */
void mult_expand_channel(Tensor<float> *Y, Tensor<float> *X1, Tensor<float> *X2)
{
    int batch_size = Y->getN();
    int channels = Y->getC();
    int plane_size = Y->plane_size();

    for (int c = 0; c < channels; c++)
    {
        for (int b = 0; b < batch_size; b++)
        {
            for (int i = 0; i < plane_size; i++)
            {
                Y->data(b, c, i) = X1->data(b, c, i) * X2->data(c);
            }
        }
    }
}

/* Y = sum(X), preserve the channel dimension */
void sum_keep_channel(Tensor<float> *Y, Tensor<float> *X)
{
    int batch_size = X->getN();
    int channels = X->getC();
    int plane_size = X->plane_size();

    for (int c = 0; c < channels; c++)
    {
        Y->data(c) = 0;
        for (int b = 0; b < batch_size; b++)
        {
            for (int i = 0; i < plane_size; i++)
            {
                Y->data(c) += X->data(b, c, i);
            }
        }
    }
}

/* Y = sum(X1 * X2), preserve the channel dimension */
void product_sum_keep_channel(Tensor<float> *Y, Tensor<float> *X1, Tensor<float> *X2)
{
    int batch_size = X1->getN();
    int channels = X1->getC();
    int plane_size = X1->plane_size();

    for (int c = 0; c < channels; c++)
    {
        Y->data(c) = 0;
        for (int b = 0; b < batch_size; b++)
        {
            for (int i = 0; i < plane_size; i++)
            {
                Y->data(c) += (X1->data(b, c, i) * X2->data(b, c, i));
            }
        }
    }
}

/* backward of batch normalization */
void backward_batchnorm(Tensor<float> *DX, Tensor<float> *DXHAT, Tensor<float> *XHAT, Tensor<float> *VAR, Tensor<float> *T1, Tensor<float> *T2)
{
    int batch_size = DX->getN();
    int channels = DX->getC();
    int plane_size = DX->plane_size();

    int N = batch_size * plane_size;

    for (int c = 0; c < channels; c++)
    {
        T1->data(c) = 0;
        T2->data(c) = 0;
        for (int b = 0; b < batch_size; b++)
        {
            for (int i = 0; i < plane_size; i++)
            {
                T1->data(c) += DXHAT->data(b, c, i);
                T2->data(c) += DXHAT->data(b, c, i) * XHAT->data(b, c, i);
            }
        }
    }

    for (int c = 0; c < channels; c++)
    {
        for (int b = 0; b < batch_size; b++)
        {
            for (int i = 0; i < plane_size; i++)
            {
                DX->data(b, c, i) = (1.0/(N*sqrt(VAR->data(c)+EPSILON))) * (DXHAT->data(b, c, i)*N - T1->data(c) - T1->data(c)*T2->data(c)*XHAT->data(b, c, i));
            }
        }
    }
}

/* calculate TD target for dqn */
void dqn_target(float *targets, float *outputs, float lambda, float *rewards, int *actions, int *final_states, int batch_size, int n)
{
    for (int i = 0; i < batch_size; i++)
    {
        int a = actions[i];
        float r = rewards[i];
        int f = final_states[i];

        float max = -FLT_MAX;
        for (int j = 0; j < n; j++)
        {
            if (targets[i*n+j] > max)
            {
                max = targets[i*n+j];
            }
        }

        for (int j = 0; j < n; j++)
        {
            targets[i*n+j] = outputs[i*n+j];
        }

        if (f)
        {
            targets[i*n+a] = 0;
        }
        else
        {
            targets[i*n+a] = max*lambda + r;
        }
    }
}
