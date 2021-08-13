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

__global__ void im2col_gpu_kernel(const int n, const float *data_im, const int input_h, const int input_w,
        const int filter_h, const int filter_w, const int padding_h, const int padding_w, const int stride_h, const int stride_w,
        const int output_h, const int output_w, float *data_col)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    for(; index < n; index += blockDim.x*gridDim.x)
    {
        int w_out = index % output_w;
        int h_index = index / output_w;
        int h_out = h_index % output_h;
        int channel_in = h_index / output_h;
        int channel_out = channel_in * filter_h * filter_w;
        int h_in = h_out * stride_h - padding_h;
        int w_in = w_out * stride_w - padding_w;
        float *data_col_ptr = data_col;
        data_col_ptr += (channel_out * output_h + h_out) * output_w + w_out;
        const float *data_im_ptr = data_im;
        data_im_ptr += (channel_in * input_h + h_in) * input_w + w_in;
        for (int i = 0; i < filter_h; i++)
        {
            for (int j = 0; j < filter_w; j++)
            {
                int h = h_in + i;
                int w = w_in + j;

                *data_col_ptr = (h >= 0 && w >= 0 && h < input_h && w < input_w) ? data_im_ptr[i * input_w + j] : 0;

                data_col_ptr += output_h * output_w;
            }
        }
    }
}

void im2col_gpu(float *data_im, int input_c, int input_h, int input_w, int output_h, int output_w,
                int filter_h, int filter_w, int stride_h, int stride_w, int padding_h, int padding_w, float *data_col)
{
    int num_kernels = input_c * output_h * output_w;
    im2col_gpu_kernel<<<(num_kernels+BLOCK-1)/BLOCK, BLOCK>>>(
                num_kernels, data_im, input_h, input_w,
                filter_h, filter_w, padding_h, padding_w, stride_h, stride_w,
                output_h, output_w, data_col);
}

__global__ void col2im_gpu_kernel(const int n, const float* data_col, const int input_h, const int input_w,
        const int filter_h, const int filter_w, const int padding_h, const int padding_w, const int stride_h, const int stride_w,
        const int output_h, const int output_w, float *data_im)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    for (; index < n; index += blockDim.x*gridDim.x)
    {
        float val = 0;
        int w = index % input_w + padding_w;
        int h = (index / input_w) % input_h + padding_h;
        int c = index / (input_w * input_h);

        int w_col_start = (w < filter_w) ? 0 : (w - filter_w) / stride_w + 1;
        int w_col_end = min(w / stride_w + 1, output_w);
        int h_col_start = (h < filter_h) ? 0 : (h - filter_h) / stride_h + 1;
        int h_col_end = min(h / stride_h + 1, output_h);

        int offset = (c * filter_h * filter_w + h * filter_w + w) * output_h * output_w;
        int coeff_h_col = (1 - stride_h * filter_h * output_h) * output_w;
        int coeff_w_col = (1 - stride_w * output_h * output_w);
        for (int h_col = h_col_start; h_col < h_col_end; h_col++)
        {
            for (int w_col = w_col_start; w_col < w_col_end; w_col++)
            {
                val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
            }
        }
        data_im[index] += val;
    }
}

void col2im_gpu(float *data_col, int input_c, int input_h, int input_w, int output_h, int output_w,
        int filter_h, int filter_w, int stride_h, int stride_w, int padding_h, int padding_w, float *data_im)
{
    int num_kernels = input_c * input_h * input_w;
    col2im_gpu_kernel<<<(num_kernels+BLOCK-1)/BLOCK,BLOCK>>>(
                num_kernels, data_col, input_h, input_w,
                filter_h, filter_w, padding_h, padding_w, stride_h, stride_w,
                output_h, output_w, data_im);
}
