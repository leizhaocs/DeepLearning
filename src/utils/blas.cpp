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

/* add biases */
void add_bias(float *output, float *biases, int batch, int n, int size)
{
    for (int b = 0; b < batch; b++)
    {
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < size; j++)
            {
                output[(b * n + i) * size + j] += biases[i];
            }
        }
    }
}

/* sum up all the elements in an array */
float sum_array(float *a, int n)
{
    float sum = 0;
    for (int i = 0; i < n; i++)
    {
        sum += a[i];
    }
    return sum;
}

/* backward of add_bias */
void backward_bias(float *grad_biases, float *delta, int batch, int n, int size)
{
    for (int b = 0; b < batch; b++)
    {
        for (int i = 0; i < n; i++)
        {
            grad_biases[i] += sum_array(delta + size * (i + b * n), size);
        }
    }
}

/* Y += ALPHA * X */
void axpy(int N, float ALPHA, float *X, float *Y)
{
    for (int i = 0; i < N; i++)
    {
        Y[i] += ALPHA * X[i];
    }
}

/* clear all elements to 0 */
void clear(int N, float *X)
{
    for (int i = 0; i < N; i++)
    {
        X[i] = 0;
    }
}

/* set all elements to random number between [0, 1] */
void random(int N, float *X)
{
    for (int i = 0; i < N; i++)
    {
        X[i] = rand()/float(RAND_MAX);
    }
}

/* copy */
void copy(float *dest, float *src, int number_of_float)
{
    memcpy(dest, src, number_of_float*sizeof(float));
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
