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

/* cross entropy loss on cpu */
void cross_entropy(Tensor<float> *errors, Tensor<float> *outputs, Tensor<float> *targets, int targets_offset, int batch_size, int classes)
{
    for (int n = 0; n < batch_size; n++)
    {
        for (int c = 0; c < classes; c++)
        {
            errors->data(n, c) = log(outputs->data(n, c)+EPSILON) * targets->data(n+targets_offset, c) * -1;
        }
    }
}

/* mean squared loss on cpu */
void mse(Tensor<float> *errors, Tensor<float> *outputs, Tensor<float> *targets, int targets_offset, int batch_size, int classes)
{
    for (int n = 0; n < batch_size; n++)
    {
        for (int c = 0; c < classes; c++)
        {
            errors->data(n, c) = outputs->data(n, c) - targets->data(n+targets_offset, c);
        }
    }
}
