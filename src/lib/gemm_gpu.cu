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
