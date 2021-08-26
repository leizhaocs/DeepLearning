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

#ifndef _POOL_H_
#define _POOL_H_

#include "includes.h"

/* max pooling */
void maxpool(Tensor<float> *input, Tensor<float> *output, Tensor<int> *index, int stride_h, int stride_w, int filter_h, int filter_w, int padding_h, int padding_w);

/* backward of maxpool */
void backward_maxpool(Tensor<float> *backward_input, Tensor<float> *backward_output, Tensor<int> *index);

#if GPU == 1
/* max pooling */
void maxpool_gpu(Tensor<float> *input, Tensor<float> *output, Tensor<int> *index, int stride_h, int stride_w, int filter_h, int filter_w, int padding_h, int padding_w);

/* backward of maxpool */
void backward_maxpool_gpu(Tensor<float> *backward_input, Tensor<float> *backward_output, Tensor<int> *index, int stride_h, int stride_w, int filter_h, int filter_w, int padding_h, int padding_w);
#endif

#endif
