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

/* max pooling */
void maxpool(Tensor<float> *input, Tensor<float> *output, Tensor<int> *index, int stride_h, int stride_w, int filter_h, int filter_w, int padding_h, int padding_w)
{
    int in_c = input->getC();
    int in_h = input->getH();
    int in_w = input->getW();
    int out_n = output->getN();
    int out_h = output->getH();
    int out_w = output->getW();

    for (int n = 0; n < out_n; n++)
    {
        for (int c = 0; c < in_c; c++)
        {
            for (int h = 0; h < out_h; h++)
            {
                for (int w = 0; w < out_w; w++)
                {
                    int start_h = h * stride_h - padding_h;
                    int start_w = w * stride_w - padding_w;
                    int end_h = start_h + filter_h;
                    int end_w = start_w + filter_w;

                    float max = -FLT_MAX;
                    int max_i = -1;
                    for (int cur_h = start_h; cur_h < end_h; cur_h++)
                    {
                        for (int cur_w = start_w; cur_w < end_w; cur_w++)
                        {
                            if (cur_h < 0 || cur_w < 0 || cur_h > in_h-1 || cur_w > in_w-1)
                            {
                            }
                            else if (input->data(n, c, cur_h, cur_w) > max)
                            {
                                max = input->data(n, c, cur_h, cur_w);
                                max_i = ((n*in_c + c)*in_h + cur_h)*in_w + cur_w;
                            }
                        }
                    }
                    output->data(n, c, h, w) = max;
                    index->data(n, c, h, w) = max_i;
                }
            }
        }
    }
}

/* backward of maxpool */
void backward_maxpool(Tensor<float> *backward_input, Tensor<float> *backward_output, Tensor<int> *index)
{
    int total_size = backward_input->total_size();

    for (int i = 0; i < total_size; i++)
    {
        int ind = index->data(i);
        backward_output->data(ind) += backward_input->data(i);
    }
}
