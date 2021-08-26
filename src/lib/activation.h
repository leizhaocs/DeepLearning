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

#ifndef _ACTIVATION_H_
#define _ACTIVATION_H_

#include "includes.h"

/* relu */
void relu(Tensor<float> *input, Tensor<float> *output);

/* backward of relu */
void backward_relu(Tensor<float> *backward_input, Tensor<float> *forward_output, Tensor<float> *backward_output);

/* sigmoid */
void sigmoid(Tensor<float> *input, Tensor<float> *output);

/* backward of sigmoid */
void backward_sigmoid(Tensor<float> *backward_input, Tensor<float> *forward_output, Tensor<float> *backward_output);

/* tanh */
void tanh(Tensor<float> *input, Tensor<float> *output);

/* backward of tanh */
void backward_tanh(Tensor<float> *backward_input, Tensor<float> *forward_output, Tensor<float> *backward_output);

/* softmax */
void softmax(Tensor<float> *input, Tensor<float> *output);

/* backward of softmax */
void backward_softmax(Tensor<float> *backward_input, Tensor<float> *forward_output, Tensor<float> *backward_output);

#if GPU == 1
/* relu */
void relu_gpu(Tensor<float> *input, Tensor<float> *output);

/* backward of relu */
void backward_relu_gpu(Tensor<float> *backward_input, Tensor<float> *forward_output, Tensor<float> *backward_output);

/* sigmoid */
void sigmoid_gpu(Tensor<float> *input, Tensor<float> *output);

/* backward of sigmoid */
void backward_sigmoid_gpu(Tensor<float> *backward_input, Tensor<float> *forward_output, Tensor<float> *backward_output);

/* tanh */
void tanh_gpu(Tensor<float> *input, Tensor<float> *output);

/* backward of tanh */
void backward_tanh_gpu(Tensor<float> *backward_input, Tensor<float> *forward_output, Tensor<float> *backward_output);

/* softmax */
void softmax_gpu(Tensor<float> *input, Tensor<float> *output);

/* backward of softmax */
void backward_softmax_gpu(Tensor<float> *backward_input, Tensor<float> *forward_output, Tensor<float> *backward_output);
#endif

#endif
