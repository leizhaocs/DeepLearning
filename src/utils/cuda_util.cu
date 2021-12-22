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

/* grid size */
dim3 cuda_gridsize(int n)
{
    int k = (n-1) / BLOCK + 1;
    int x = k;
    int y = 1;
    if (x > 65535)
    {
        x = ceil(sqrt(k));
        y = (n-1)/(x*BLOCK) + 1;
    }
    dim3 d = {(unsigned int)x, (unsigned int)y, 1};
    return d;
}

/* get cublas handle */
cublasHandle_t cublas_handle()
{
    static int cublas_init = 0;
    static cublasHandle_t cublas_handle;
    if (!cublas_init)
    {
        cublasCreate(&cublas_handle);
        cublas_init = 1;
    }
    return cublas_handle;
}

#if CUDNN == 1
/* get cudnn handle */
cudnnHandle_t cudnn_handle()
{
    static int cudnn_init = 0;
    static cudnnHandle_t cudnn_handle;
    if (!cudnn_init)
    {
        CHECK_CUDNN_ERRORS(cudnnCreate(&cudnn_handle));
        cudnn_init = 1;
    }
    return cudnn_handle;
}
#endif
