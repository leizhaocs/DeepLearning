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

void gemm_nn(int M, int N, int K, float *A, int lda, float *B, int ldb, float *C, int ldc)
{
    #pragma omp parallel for
    for (int i = 0; i < M; i++)
    {
        for (int k = 0; k < K; k++)
        {
            float A_PART = A[i * lda + k];
            for (int j = 0; j < N; j++)
            {
                C[i * ldc + j] += A_PART * B[k * ldb + j];
            }
        }
    }
}

void gemm_nt(int M, int N, int K, float *A, int lda, float *B, int ldb, float *C, int ldc)
{
    #pragma omp parallel for
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float sum = 0;
            for (int k = 0; k < K; k++)
            {
                sum += A[i * lda + k] * B[j * ldb + k];
            }
            C[i * ldc + j] += sum;
        }
    }
}

void gemm_tn(int M, int N, int K, float *A, int lda, float *B, int ldb, float *C, int ldc)
{
    #pragma omp parallel for
    for (int i = 0; i < M; i++)
    {
        for (int k = 0; k < K; k++)
        {
            float A_PART = A[k * lda + i];
            for (int j = 0; j < N; j++)
            {
                C[i * ldc + j] += A_PART * B[k * ldb + j];
            }
        }
    }
}

void gemm_tt(int M, int N, int K, float *A, int lda, float *B, int ldb, float *C, int ldc)
{
    #pragma omp parallel for
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float sum = 0;
            for (int k = 0; k < K; k++)
            {
                sum += A[i + k * lda] * B[k + j * ldb];
            }
            C[i * ldc + j] += sum;
        }
    }
}

/* TA: transpose for A;    TB: transpose for B
 * A(M*K) X B(K*N) = C(M*N)
 * lda ldb and ldc are the lengths of the last dimension of A B and C
 * e.g.  gemm(0, 0, m, n, k, A, k, B, n, C, n)
 *       gemm(0, 1, m, n, k, A, k, B, k, C, n)
 *       gemm(1, 0, m, n, k, A, m, B, n, C, n)
 *       gemm(1, 1, m, n, k, A, m, B, k, C, n)
 */
void gemm(int TA, int TB, int M, int N, int K, float *A, int lda, float *B, int ldb, float *C, int ldc)
{
    if (!TA && !TB)
    {
        gemm_nn(M, N, K, A, lda, B, ldb, C, ldc);
    }
    else if (TA && !TB)
    {
        gemm_tn(M, N, K, A, lda, B, ldb, C, ldc);
    }
    else if (!TA && TB)
    {
        gemm_nt(M, N, K, A, lda, B, ldb, C, ldc);
    }
    else
    {
        gemm_tt(M, N, K, A, lda, B, ldb, C, ldc);
    }
}

#if GPU == 1
/* TA: transpose for A;    TB: transpose for B
 * A(M*K) X B(K*N) = C(M*N)
 * lda ldb and ldc are the lengths of the last dimension of A B and C
 * e.g.  gemm(0, 0, m, n, k, A, k, B, n, C, n)
 *       gemm(0, 1, m, n, k, A, k, B, k, C, n)
 *       gemm(1, 0, m, n, k, A, m, B, n, C, n)
 *       gemm(1, 1, m, n, k, A, m, B, k, C, n)
 */
void gemm_gpu(int TA, int TB, int M, int N, int K, float *A_gpu, int lda, float *B_gpu, int ldb, float *C_gpu, int ldc)
{
    float ALPHA_BETA = 1;

    cublasHandle_t handle = blas_handle();
    cublasStatus_t status = cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N), (TA ? CUBLAS_OP_T : CUBLAS_OP_N),
            N, M, K, &ALPHA_BETA, B_gpu, ldb, A_gpu, lda, &ALPHA_BETA, C_gpu, ldc);
    check_cublas_error(status);
}
#endif
