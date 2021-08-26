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

#ifndef _CONV_H_
#define _CONV_H_

#include "includes.h"

/* im2col before conv */
void im2col_cpu(float* data_im, int input_c, int input_h, int input_w, int output_h, int output_w,
                int filter_h, int filter_w, int stride_h, int stride_w, int padding_h, int padding_w, float* data_col);

/* col2im after conv */
void col2im_cpu(float *data_col, int input_c, int input_h, int input_w, int output_h, int output_w,
                int filter_h, int filter_w, int stride_h, int stride_w, int padding_h, int padding_w, float *data_im);

#if GPU == 1
/* im2col before conv */
void im2col_gpu(float *data_im, int input_c, int input_h, int input_w, int output_h, int output_w,
                int filter_h, int filter_w, int stride_h, int stride_w, int padding_h, int padding_w, float *data_col);

/* col2im after conv */
void col2im_gpu(float *data_col, int input_c, int input_h, int input_w, int output_h, int output_w,
                int filter_h, int filter_w, int stride_h, int stride_w, int padding_h, int padding_w, float *data_im);
#endif

#endif
