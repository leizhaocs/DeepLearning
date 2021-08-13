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

float im2col_get_pixel(float *im, int input_h, int input_w, int h, int w, int c, int padding_h, int padding_w)
{
    h -= padding_h;
    w -= padding_w;

    if (h < 0 || w < 0 || h >= input_h || w >= input_w)
    {
        return 0;
    }
    return im[(c*input_h + h)*input_w + w];
}

void im2col_cpu(float* data_im, int input_c, int input_h, int input_w, int output_h, int output_w,
                int filter_h, int filter_w, int stride_h, int stride_w, int padding_h, int padding_w, float* data_col) 
{
    int one_filter_sz = input_c * filter_h * filter_w;

    for (int f = 0; f < one_filter_sz; f++)
    {
        int kernel_w_offset = f % filter_w;
        int kernel_h_offset = (f / filter_w) % filter_h;
        int kernel_c_offset = f / filter_w / filter_h;

        for (int h = 0; h < output_h; h++)
        {
            for (int w = 0; w < output_w; w++)
            {
                int im_h = kernel_h_offset + h * stride_h;
                int im_w = kernel_w_offset + w * stride_w;

                int index = (f * output_h + h) * output_w + w;
                data_col[index] = im2col_get_pixel(data_im, input_h, input_w, im_h, im_w, kernel_c_offset, padding_h, padding_w);
            }
        }
    }
}

void col2im_add_pixel(float *im, int input_h, int input_w, int h, int w, int c, int padding_h, int padding_w, float val)
{
    h -= padding_h;
    w -= padding_w;

    if (h < 0 || w < 0 || h >= input_h || w >= input_w)
    {
        return;
    }
    im[(c*input_h + h)*input_w + w] += val;
}

void col2im_cpu(float *data_col, int input_c, int input_h, int input_w, int output_h, int output_w,
                int filter_h, int filter_w, int stride_h, int stride_w, int padding_h, int padding_w, float *data_im) 
{
    int one_filter_sz = input_c * filter_h * filter_w;

    for (int f = 0; f < one_filter_sz; f++)
    {
        int kernel_w_offset = f % filter_w;
        int kernel_h_offset = (f / filter_w) % filter_h;
        int kernel_c_offset = f / filter_w / filter_h;

        for (int h = 0; h < output_h; h++)
        {
            for (int w = 0; w < output_w; w++)
            {
                int im_h = kernel_h_offset + h * stride_h;
                int im_w = kernel_w_offset + w * stride_w;

                int index = (f * output_h + h) * output_w + w;
                float val = data_col[index];
                col2im_add_pixel(data_im, input_h, input_w, im_h, im_w, kernel_c_offset, padding_h, padding_w, val);
            }
        }
    }
}
