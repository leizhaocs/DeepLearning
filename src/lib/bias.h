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

#ifndef _BIAS_H_
#define _BIAS_H_

#include "includes.h"

/* backward of add_bias */
void backward_bias(Tensor<float> *grad_biases, Tensor<float> *delta);

#if GPU == 1
/* calculate gradient of biases */
void backward_bias_gpu(Tensor<float> *grad_biases, Tensor<float> *delta);
#endif

#endif
