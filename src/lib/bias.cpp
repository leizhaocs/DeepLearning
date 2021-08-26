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
void backward_bias(Tensor<float> *grad_biases, Tensor<float> *delta)
{
    int batch_size = delta->getN();
    int channels = delta->getC();
    int plane_size = delta->plane_size();

    for (int b = 0; b < batch_size; b++)
    {
        for (int c = 0; c < channels; c++)
        {
            grad_biases->data(c) += sum_array(delta->getCpuPtr() + plane_size * (c + b * channels), plane_size);
        }
    }
}
