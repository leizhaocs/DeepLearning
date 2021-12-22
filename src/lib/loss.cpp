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

/* FIXME: correctly calculate the loss */
/* cross entropy loss on cpu */
void cross_entropy(Tensor<float> *errors, Tensor<float> *outputs, Tensor<float> *targets, int targets_offset)
{
    int batch_size = errors->getN();
    int classes = errors->sample_size();

    for (int b = 0; b < batch_size; b++)
    {
        for (int c = 0; c < classes; c++)
        {
            errors->data(b, c) = log(outputs->data(b, c)+EPSILON) * targets->data(b+targets_offset, c) * -1;
        }
    }
}

/* backward of cross entropy loss */
void backward_cross_entropy(Tensor<float> *errors, Tensor<float> *outputs, Tensor<float> *targets, int targets_offset)
{
    int batch_size = errors->getN();
    int classes = errors->sample_size();

    for (int b = 0; b < batch_size; b++)
    {
        for (int c = 0; c < classes; c++)
        {
            if (targets->data(b+targets_offset, c) != 0)
            {
                errors->data(b, c) = -1 / (outputs->data(b, c)+EPSILON);
            }
            else
            {
                errors->data(b, c) = 0;
            }
        }
    }
}

/* FIXME: correctly calculate the loss */
/* mean squared loss on cpu */
void mse(Tensor<float> *errors, Tensor<float> *outputs, Tensor<float> *targets, int targets_offset)
{
    int batch_size = errors->getN();
    int classes = errors->sample_size();

    for (int b = 0; b < batch_size; b++)
    {
        for (int c = 0; c < classes; c++)
        {
            errors->data(b, c) = outputs->data(b, c) - targets->data(b+targets_offset, c);
        }
    }
}

/* backward of mean squared loss */
void backward_mse(Tensor<float> *errors, Tensor<float> *outputs, Tensor<float> *targets, int targets_offset)
{
    int batch_size = errors->getN();
    int classes = errors->sample_size();

    for (int b = 0; b < batch_size; b++)
    {
        for (int c = 0; c < classes; c++)
        {
            errors->data(b, c) = outputs->data(b, c) - targets->data(b+targets_offset, c);
        }
    }
}
