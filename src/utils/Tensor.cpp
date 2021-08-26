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

/* constructor */
template<typename T>
Tensor<T>::Tensor(int n, int c, int h, int w)
{
    n_ = n;
    c_ = c;
    h_ = h;
    w_ = w;
    total_size_ = n * c * h * w;
    sample_size_ = c * h * w;
    plane_size_ = h * w;

    MemoryMonitor::instance()->cpuMalloc((void**)&data_cpu_, total_size_*sizeof(T));
#if GPU == 1
    if (use_gpu)
    {
        MemoryMonitor::instance()->gpuMalloc((void**)&data_gpu_, total_size_*sizeof(T));
    }
#endif
}

/* destructor */
template<typename T>
Tensor<T>::~Tensor()
{
    MemoryMonitor::instance()->freeCpuMemory(data_cpu_);
#if GPU == 1
    if (use_gpu)
    {
        MemoryMonitor::instance()->freeGpuMemory(data_gpu_);
    }
#endif
}

/* get cpu data pointer */
template<typename T>
T *Tensor<T>::getCpuPtr()
{
    return data_cpu_;
}

#if GPU == 1
/* get gpu data pointer */
template<typename T>
T *Tensor<T>::getGpuPtr()
{
    return data_gpu_;
}
#endif

/* get dimensions */
template<typename T>
int Tensor<T>::getN()
{
    return n_;
}
template<typename T>
int Tensor<T>::getC()
{
    return c_;
}
template<typename T>
int Tensor<T>::getH()
{
    return h_;
}
template<typename T>
int Tensor<T>::getW()
{
    return w_;
}

/* get total number of elements */
template<typename T>
int Tensor<T>::total_size()
{
    return total_size_;
}

/* get number of elements in one sample */
template<typename T>
int Tensor<T>::sample_size()
{
    return sample_size_;
}

/* get number of elements in one plane */
template<typename T>
int Tensor<T>::plane_size()
{
    return plane_size_;
}

/* get data element */
template<typename T>
T &Tensor<T>::data(int i)
{
    return data_cpu_[i];
}
template<typename T>
T &Tensor<T>::data(int n, int i)
{
    int index = n*sample_size_ + i;
    return data_cpu_[index];
}
template<typename T>
T &Tensor<T>::data(int n, int c, int i)
{
    int index = n*sample_size_ + c*plane_size_ + i;
    return data_cpu_[index];
}
template<typename T>
T &Tensor<T>::data(int n, int c, int h, int w)
{
    int index = n*sample_size_ + c*plane_size_ + h*w_ + w;
    return data_cpu_[index];
}

#if GPU == 1
/* move data from cpu to gpu */
template<typename T>
void Tensor<T>::toGpu()
{
    cudaError_t cudaStat = cudaMemcpy(data_gpu_, data_cpu_, total_size_*sizeof(T), cudaMemcpyHostToDevice);
    Assert(cudaStat == cudaSuccess, "To gpu data upload failed.");
}

/* move data from gpu to cpu */
template<typename T>
void Tensor<T>::toCpu()
{
    cudaError_t cudaStat = cudaMemcpy(data_cpu_, data_gpu_, total_size_*sizeof(T), cudaMemcpyDeviceToHost);
    Assert(cudaStat == cudaSuccess, "To cpu data download failed.");
}
#endif

template class Tensor<float>;
template class Tensor<int>;
