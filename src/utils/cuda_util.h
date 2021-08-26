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

#ifndef __CUDA_UTIL__
#define __CUDA_UTIL__

#include "includes.h"

#if GPU == 1
/* grid size */
dim3 cuda_gridsize(int n);

/* check cuda error */
void check_cuda_error();

/* check cublas error */
void check_cublas_error(cublasStatus_t status);

/* get blas handle */
cublasHandle_t blas_handle();
#endif

#endif
