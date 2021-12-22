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
#define CHECK_CUDA_ERRORS()                                                                               \
{                                                                                                         \
    cudaError_t err = cudaGetLastError();                                                                 \
    if (err != cudaSuccess)                                                                               \
    {                                                                                                     \
        printf("cuda failure: %s, in file <%s>, line %d.\n", cudaGetErrorString(err), __FILE__, __LINE__);\
        exit(-1);                                                                                         \
    }                                                                                                     \
}

/* cublas error type string */
static const char *cublas_error_string(cublasStatus_t status)
{
    switch (status)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "<unknown>";
}

/* check cublas error */
#define CHECK_CUBLAS_ERRORS(err)                                                                             \
{                                                                                                            \
    if (err != CUBLAS_STATUS_SUCCESS)                                                                        \
    {                                                                                                        \
        printf("cublas failure: %s, in file <%s>, line %d.\n", cublas_error_string(err), __FILE__, __LINE__);\
        exit(-1);                                                                                            \
    }                                                                                                        \
}

/* get cublas handle */
cublasHandle_t cublas_handle();

#if CUDNN == 1
/* check cudnn error */
#define CHECK_CUDNN_ERRORS(err)                                                                             \
{                                                                                                           \
    if (err != CUDNN_STATUS_SUCCESS)                                                                        \
    {                                                                                                       \
        printf("cudnn failure: %s, in file <%s>, line %d.\n", cudnnGetErrorString(err), __FILE__, __LINE__);\
        exit(-1);                                                                                           \
    }                                                                                                       \
}

/* get cudnn handle */
cudnnHandle_t cudnn_handle();
#endif
#endif

#endif
