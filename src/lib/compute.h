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

#ifndef _MATH_H_
#define _MATH_H_

#include "includes.h"

/* X = 0 */
void clear(Tensor<float> *X);

/* X = rand(0, 1) */
void random(Tensor<float> *X);

/* Y += X, expand X's channel dimension */
void add_expand_channel(Tensor<float> *Y, Tensor<float> *X);

/* Y += ALPHA * X */
void axpy(Tensor<float> *Y, Tensor<float> *X, float ALPHA);

/* Y = X */
void assign(Tensor<float> *Y, Tensor<float> *X);

/* Y = (R < S) ? 0 : X */
void assign_cond(Tensor<float> *Y, Tensor<float> *X, Tensor<float> *R, float S);

/* Y = mean(X), preserve the channel dimension */
void mean_keep_channel(Tensor<float> *Y, Tensor<float> *X);

/* Y = variance(X, MEAN), preserve the channel dimension */
void variance_keep_channel(Tensor<float> *Y, Tensor<float> *X, Tensor<float> *MEAN);

/* Y = normalize(X, MEAN, VAR), expand MEAN's and VAR's channel dimension */
void normalize_expand_channel(Tensor<float> *Y, Tensor<float> *X, Tensor<float> *MEAN, Tensor<float> *VAR);

/* Y = GAMMA * X + BETA, expand GAMMA's and BETA's channel dimension */
void scale_shift_expand_channel(Tensor<float> *Y, Tensor<float> *X, Tensor<float> *GAMMA, Tensor<float> *BETA);

/* Y = Y * M + X * (1 - M) */
void add_with_momentum(Tensor<float> *Y, Tensor<float> *X, float M);

/* Y = X1 * X2, expand X2's channel dimension */
void mult_expand_channel(Tensor<float> *Y, Tensor<float> *X1, Tensor<float> *X2);

/* Y = sum(X), preserve the channel dimension */
void sum_keep_channel(Tensor<float> *Y, Tensor<float> *X);

/* Y = sum(X1 * X2), preserve the channel dimension */
void product_sum_keep_channel(Tensor<float> *Y, Tensor<float> *X1, Tensor<float> *X2);

/* backward of batch normalization */
void backward_batchnorm(Tensor<float> *DX, Tensor<float> *DXHAT, Tensor<float> *XHAT, Tensor<float> *VAR, Tensor<float> *T1, Tensor<float> *T2);

/* calculate target for dqn */
void dqn_target(float *targets, float *outputs, float lambda, float *rewards, int *actions, int *final_states, int batch_size, int n);

#if GPU == 1
/* X = 0 */
void clear_gpu(Tensor<float> *X);

/* X = rand(0, 1) */
void random_gpu(Tensor<float> *X);

/* Y += X, expand X's channel dimension */
void add_expand_channel_gpu(Tensor<float> *Y, Tensor<float> *X);

/* Y += ALPHA * X */
void axpy_gpu(Tensor<float> *Y, Tensor<float> *X, float ALPHA);

/* Y = X */
void assign_gpu(Tensor<float> *Y, Tensor<float> *X);

/* Y = (R < S) ? 0 : X */
void assign_cond_gpu(Tensor<float> *Y, Tensor<float> *X, Tensor<float> *R, float S);

/* Y = mean(X), preserve the channel dimension */
void mean_keep_channel_gpu(Tensor<float> *Y, Tensor<float> *X);

/* Y = variance(X, MEAN), preserve the channel dimension */
void variance_keep_channel_gpu(Tensor<float> *Y, Tensor<float> *X, Tensor<float> *MEAN);

/* Y = normalize(X, MEAN, VAR), expand MEAN's and VAR's channel dimension */
void normalize_expand_channel_gpu(Tensor<float> *Y, Tensor<float> *X, Tensor<float> *MEAN, Tensor<float> *VAR);

/* Y = GAMMA * X + BETA, expand GAMMA's and BETA's channel dimension */
void scale_shift_expand_channel_gpu(Tensor<float> *Y, Tensor<float> *X, Tensor<float> *GAMMA, Tensor<float> *BETA);

/* Y = Y * M + X * (1 - M) */
void add_with_momentum_gpu(Tensor<float> *Y, Tensor<float> *X, float M);

/* Y = X1 * X2, expand X2's channel dimension */
void mult_expand_channel_gpu(Tensor<float> *Y, Tensor<float> *X1, Tensor<float> *X2);

/* Y = sum(X), preserve the channel dimension */
void sum_keep_channel_gpu(Tensor<float> *Y, Tensor<float> *X);

/* Y = sum(X1 * X2), preserve the channel dimension */
void product_sum_keep_channel_gpu(Tensor<float> *Y, Tensor<float> *X1, Tensor<float> *X2);

/* backward of batch normalization */
void backward_batchnorm_gpu(Tensor<float> *DX, Tensor<float> *DXHAT, Tensor<float> *XHAT, Tensor<float> *VAR, Tensor<float> *T1, Tensor<float> *T2);

/* calculate TD target for dqn */
void dqn_target_gpu(float *targets, float *outputs, float lambda, float *rewards, int *actions, int *final_states, int batch_size, int n);
#endif

#endif
