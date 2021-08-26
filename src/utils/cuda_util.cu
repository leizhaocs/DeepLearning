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

/* check cuda error */
void check_cuda_error()
{                     
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("Cuda failure %s:%d:\n",__FILE__,__LINE__);
        exit(1);
    }
}

/* check cublas error */
void check_cublas_error(cublasStatus_t status)
{                     
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        printf("Cuda failure %s:%d\n",__FILE__,__LINE__);
        exit(1);
    }
}

/* get blas handle */
cublasHandle_t blas_handle()
{
    static int init = 0;
    static cublasHandle_t handle;
    if (!init)
    {
        cublasCreate(&handle);
        init = 1;
    }
    return handle;
}
