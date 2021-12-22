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
#if CUDNN == 1
    init_tensor_desc_ = false;
    init_filter_desc_ = false;
#endif
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
#if CUDNN == 1
    if (init_tensor_desc_)
    {
        CHECK_CUDNN_ERRORS(cudnnDestroyTensorDescriptor(tensor_desc_));
    }
    if (init_filter_desc_)
    {
        CHECK_CUDNN_ERRORS(cudnnDestroyFilterDescriptor(filter_desc_));
    }
#endif
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

#if CUDNN == 1
/* get the tensor descriptor */
template<typename T>
cudnnTensorDescriptor_t Tensor<T>::getTensorDescriptor()
{
    if (init_tensor_desc_ == false)
    {
        CHECK_CUDNN_ERRORS(cudnnCreateTensorDescriptor(&tensor_desc_));
        if (strcmp(typeid(data_gpu_).name(), "Pf") == 0)
        {
            CHECK_CUDNN_ERRORS(cudnnSetTensor4dDescriptor(tensor_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n_, c_, h_, w_));
        }
        else if (strcmp(typeid(data_gpu_).name(), "Pi") == 0)
        {
            CHECK_CUDNN_ERRORS(cudnnSetTensor4dDescriptor(tensor_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_INT32, n_, c_, h_, w_));
        }
        else
        {
            Assert(false, "Unsupported data type in cudnn tensor.");
        }
        init_tensor_desc_ = true;
    }
    return tensor_desc_;
}

/* get the filter descriptor */
template<typename T>
cudnnFilterDescriptor_t Tensor<T>::getFilterDescriptor()
{
    if (init_filter_desc_ == false)
    {
        CHECK_CUDNN_ERRORS(cudnnCreateFilterDescriptor(&filter_desc_));
        if (strcmp(typeid(data_gpu_).name(), "Pf") == 0)
        {
            CHECK_CUDNN_ERRORS(cudnnSetFilter4dDescriptor(filter_desc_, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n_, c_, h_, w_));
        }
        else
        {
            Assert(false, "Unsupported data type in cudnn filter.");
        }
        init_filter_desc_ = true;
    }
    return filter_desc_;
}
#endif
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

/* get the total memory space of the tensor */
template<typename T>
int Tensor<T>::total_bytes()
{
    return total_size_ * sizeof(T);
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
    if (use_gpu)
    {
        cudaMemcpy(data_gpu_, data_cpu_, total_size_*sizeof(T), cudaMemcpyHostToDevice);
        CHECK_CUDA_ERRORS();
    }
}

/* move data from gpu to cpu */
template<typename T>
void Tensor<T>::toCpu()
{
    if (use_gpu)
    {
        cudaMemcpy(data_cpu_, data_gpu_, total_size_*sizeof(T), cudaMemcpyDeviceToHost);
        CHECK_CUDA_ERRORS();
    }
}
#endif

template class Tensor<float>;
template class Tensor<int>;
