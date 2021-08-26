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

/* relu */
void relu(Tensor<float> *input, Tensor<float> *output)
{
    int total_size = output->total_size();

    for (int i = 0; i < total_size; i++)
    {
        output->data(i) = (input->data(i) > 0) ? input->data(i) : 0;
    }
}

/* backward of relu */
void backward_relu(Tensor<float> *backward_input, Tensor<float> *forward_output, Tensor<float> *backward_output)
{
    int total_size = backward_output->total_size();

    for (int i = 0; i < total_size; i++)
    {
        backward_output->data(i) = (forward_output->data(i) > 0) ? backward_input->data(i) : 0;
    }
}

/* sigmoid */
void sigmoid(Tensor<float> *input, Tensor<float> *output)
{
    int total_size = output->total_size();

    for (int i = 0; i < total_size; i++)
    {
        float temp = exp((float)input->data(i));
        output->data(i) = temp / (temp + 1);
    }
}

/* backward of sigmoid */
void backward_sigmoid(Tensor<float> *backward_input, Tensor<float> *forward_output, Tensor<float> *backward_output)
{
    int total_size = backward_output->total_size();

    for (int i = 0; i < total_size; i++)
    {
        backward_output->data(i) = backward_input->data(i) * forward_output->data(i) * (1 - forward_output->data(i));
    }
}

/* tanh */
void tanh(Tensor<float> *input, Tensor<float> *output)
{
    int total_size = output->total_size();

    for (int i = 0; i < total_size; i++)
    {
        float temp = exp((float)input->data(i));
        float temp_inverse = 1 / temp;
        output->data(i) = (temp - temp_inverse) / (temp + temp_inverse);
    }
}

/* backward of tanh */
void backward_tanh(Tensor<float> *backward_input, Tensor<float> *forward_output, Tensor<float> *backward_output)
{
    int total_size = backward_output->total_size();

    for (int i = 0; i < total_size; i++)
    {
        backward_output->data(i) = backward_input->data(i) * (1 - forward_output->data(i) * forward_output->data(i));
    }
}

/* softmax */
void softmax(Tensor<float> *input, Tensor<float> *output)
{
    int batch_size = output->getN();
    int sample_size = output->sample_size();

    for (int b = 0; b < batch_size; b++)
    {
        float max = -FLT_MAX;
        for (int s = 0; s < sample_size; s++)
        {
            if (input->data(b, s) > max)
            {
                max = input->data(b, s);
            }
        }
        float sum = 0;
        for (int s = 0; s < sample_size; s++)
        {
            output->data(b, s) = exp((float)(input->data(b, s)-max));
            sum += output->data(b, s);
        }
        for (int s = 0; s < sample_size; s++)
        {
            output->data(b, s) /= sum;
        }
    }
}

/* backward of softmax */
void backward_softmax(Tensor<float> *backward_input, Tensor<float> *forward_output, Tensor<float> *backward_output)
{
    int total_size = backward_output->total_size();

    for (int i = 0; i < total_size; i++)
    {
        backward_output->data(i) = forward_output->data(i) - (backward_input->data(i) != 0);
    }
}
