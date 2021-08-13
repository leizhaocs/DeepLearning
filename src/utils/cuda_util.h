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

#ifndef __CUDA_UTIL__
#define __CUDA_UTIL__

#include "includes.h"

/* dropout */
void dropout_gpu(float *input, float *output, int n, int batch, float *rand, float rate);

/* backward of dropout */
void backward_dropout_gpu(float *backward_input, float *backward_output, float *rand, float rate, int n, int batch);

/* max pooling */
void maxpool_gpu(float *input, float *output, int *index, int in_h, int in_w, int in_c, int out_h, int out_w,
    int stride_h, int stride_w, int filter_h, int filter_w, int padding_h, int padding_w, int batch);

/* backward of maxpool */
void backward_maxpool_gpu(float *backward_input, float *backward_output, int *index, int in_h, int in_w, int in_c, int out_h, int out_w,
    int stride_h, int stride_w, int filter_h, int filter_w, int padding_h, int padding_w, int batch);

/* relu */
void relu_gpu(float *input, float *output, int n);

/* backward of relu */
void backward_relu_gpu(float *backward_input, float *forward_output, float *backward_output, int n);

/* sigmoid */
void sigmoid_gpu(float *input, float *output, int n);

/* backward of sigmoid */
void backward_sigmoid_gpu(float *backward_input, float *forward_output, float *backward_output, int n);

/* softmax */
void softmax_gpu(float *input, float *output, int n, int batch);

/* backward of softmax */
void backward_softmax_gpu(float *backward_input, float *forward_output, float *backward_output, int n, int batch);

/* grid size */
dim3 cuda_gridsize(int n);

/* check cuda error */
void check_cuda_error();

/* check cublas error */
void check_cublas_error(cublasStatus_t status);

/* get blas handle */
cublasHandle_t blas_handle();

#endif
