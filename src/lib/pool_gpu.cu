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

__global__ void maxpool_kernel(int N, int in_h, int in_w, int in_c, int out_h, int out_w,
    int stride_h, int stride_w, int filter_h, int filter_w, int padding_h, int padding_w,
    float *input, float *output, int *index)
{
    int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= N)
    {
        return;
    }

    int w = id % out_w;
    id /= out_w;
    int h = id % out_h;
    id /= out_h;
    int c = id % in_c;
    id /= in_c;
    int n = id;

    float max = -INFINITY;
    int max_i = -1;
    for (int i = 0; i < filter_h; i++)
    {
        for (int j = 0; j < filter_w; j++)
        {
            int cur_h = h * stride_h + i - padding_h;
            int cur_w = w * stride_w + j - padding_w;

            int in_index = ((n*in_c + c)*in_h + cur_h)*in_w + cur_w;
            int valid = (cur_h >= 0 && cur_h < in_h && cur_w >= 0 && cur_w < in_w);
            float val = (valid != 0) ? input[in_index] : -INFINITY;
            max_i = (val > max) ? in_index : max_i;
            max = (val > max) ? val : max;
        }
    }

    int out_index = ((n*in_c + c)*out_h+ h)*out_w + w;
    output[out_index] = max;
    index[out_index] = max_i;
}

/* max pooling */
void maxpool_gpu(Tensor<float> *input, Tensor<float> *output, Tensor<int> *index, int stride_h, int stride_w, int filter_h, int filter_w, int padding_h, int padding_w)
{
    float *input_ptr = input->getGpuPtr();
    float *output_ptr = output->getGpuPtr();
    int *index_ptr = index->getGpuPtr();
    int in_c = input->getC();
    int in_h = input->getH();
    int in_w = input->getW();
    int out_n = output->getN();
    int out_h = output->getH();
    int out_w = output->getW();
    int N = out_h * out_w * in_c * out_n;

    maxpool_kernel<<<GRID(N), BLOCK>>>(N, in_h, in_w, in_c, out_h, out_w,
        stride_h, stride_w, filter_h, filter_w, padding_h, padding_w,
        input_ptr, output_ptr, index_ptr);
    CHECK_CUDA_ERRORS();
}

__global__ void backward_maxpool_kernel(int N, int in_h, int in_w, int in_c, int out_h, int out_w,
    int stride_h, int stride_w, int filter_h, int filter_w, int padding_h, int padding_w,
    float *backward_input, float *backward_output, int *index)
{
    int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= N)
    {
        return;
    }

    int in_index = id;
    int w = id % in_w;
    id /= in_w;
    int h = id % in_h;
    id /= in_h;
    int c = id % in_c;
    id /= in_c;
    int n = id;

    float d = 0;
    int area_h = (filter_h - 1) / stride_h;
    int area_w = (filter_w - 1) / stride_w;
    for (int l = -area_h; l < area_h + 1; l++)
    {
        for (int m = -area_w; m < area_w + 1; m++)
        {
            int cur_h = (h + padding_h) / stride_h + l;
            int cur_w = (w + padding_w) / stride_w + m;

            int out_index = ((n*in_c + c)*out_h + cur_h)*out_w + cur_w;
            int valid = (cur_w >= 0 && cur_w < out_w && cur_h >= 0 && cur_h < out_h);
            if (valid && index[out_index] == in_index)
            {
                d += backward_input[out_index];
            }
        }
    }
    backward_output[in_index] += d;
}

/* backward of maxpool */
void backward_maxpool_gpu(Tensor<float> *backward_input, Tensor<float> *backward_output, Tensor<int> *index, int stride_h, int stride_w, int filter_h, int filter_w, int padding_h, int padding_w)
{
    float *backward_input_ptr = backward_input->getGpuPtr();
    float *backward_output_ptr = backward_output->getGpuPtr();
    int *index_ptr = index->getGpuPtr();
    int in_c = backward_output->getC();
    int in_h = backward_output->getH();
    int in_w = backward_output->getW();
    int out_n = backward_input->getN();
    int out_h = backward_input->getH();
    int out_w = backward_input->getW();
    int N = in_c * in_h * in_w * out_n;

    backward_maxpool_kernel<<<GRID(N), BLOCK>>>(N, in_h, in_w, in_c, out_h, out_w,
        stride_h, stride_w, filter_h, filter_w, padding_h, padding_w,
        backward_input_ptr, backward_output_ptr, index_ptr);
    CHECK_CUDA_ERRORS();
}
