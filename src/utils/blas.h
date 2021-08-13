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

#ifndef _BLAS_H_
#define _BLAS_H_

#include "includes.h"

/* add biases */
void add_bias(float *output, float *biases, int batch, int n, int size);

/* backward of add_bias */
void backward_bias(float *grad_biases, float *delta, int batch, int n, int size);

/* Y += ALPHA * X */
void axpy(int N, float ALPHA, float *X, float *Y);

/* clear all elements to 0 */
void clear(int N, float *X);

/* set all elements to random number between [0, 1] */
void random(int N, float *X);

/* copy */
void copy(float *dest, float *src, int number_of_float);

/* calculate target for dqn */
void dqn_target(float *targets, float *outputs, float lambda, float *rewards, int *actions, int *final_states, int batch_size, int n);

#if GPU == 1
/* add biases in gpu */
void add_bias_gpu(float *output, float *biases, int batch, int n, int size);

/* calculate gradient of biases in gpu */
void backward_bias_gpu(float *grad_biases, float *delta, int batch, int n, int size);

/* Y += ALPHA * X */
void axpy_gpu(int N, float ALPHA, float *X, float *Y);

/* clear all elements to 0 */
void clear_gpu(int N, float *X);

/* set all elements to random number between [0, 1] */
void random_gpu(int N, float *X);

/* copy */
void copy_gpu(float *dest, float *src, int number_of_float);

/* calculate TD target for dqn */
void dqn_target_gpu(float *targets, float *outputs, float lambda, float *rewards, int *actions, int *final_states, int batch_size, int n);
#endif

#endif
