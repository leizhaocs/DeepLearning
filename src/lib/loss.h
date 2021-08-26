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

#ifndef _LOSS_H__
#define _LOSS_H__

#include "includes.h"

/* cross entropy loss */
void cross_entropy(Tensor<float> *errors, Tensor<float> *outputs, Tensor<float> *targets, int targets_offset);

/* mean squared loss */
void mse(Tensor<float> *errors, Tensor<float> *outputs, Tensor<float> *targets, int targets_offset);

#if GPU == 1
/* cross entropy loss */
void cross_entropy_gpu(Tensor<float> *errors, Tensor<float> *outputs, Tensor<float> *targets, int targets_offset);

/* mean squared loss */
void mse_gpu(Tensor<float> *errors, Tensor<float> *outputs, Tensor<float> *targets, int targets_offset);
#endif

#endif
