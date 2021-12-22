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

#ifndef _TENSOR_H_
#define _TENSOR_H_

#include "includes.h"

/* tensor */
template<typename T>
class Tensor
{
public:
    /* constructor */
    Tensor(int n, int c, int h, int w);

    /* destructor */
    ~Tensor();

    /* get cpu data pointer */
    T *getCpuPtr();

#if GPU == 1
    /* get gpu data pointer */
    T *getGpuPtr();

#if CUDNN == 1
    /* get the tensor descriptor */
    cudnnTensorDescriptor_t getTensorDescriptor();

    /* get the filter descriptor */
    cudnnFilterDescriptor_t getFilterDescriptor();
#endif
#endif

    /* get dimensions */
    int getN();
    int getC();
    int getH();
    int getW();

    /* get total number of elements */
    int total_size();

    /* get number of elements in one sample */
    int sample_size();

    /* get number of elements in one plane */
    int plane_size();

    /* get the total memory space of the tensor */
    int total_bytes();

    /* get data element */
    T &data(int i);
    T &data(int n, int i);
    T &data(int n, int c, int i);
    T &data(int n, int c, int h, int w);

#if GPU == 1
    /* move data from cpu to gpu */
    void toGpu();

    /* move data from gpu to cpu */
    void toCpu();
#endif

private:
    T *data_cpu_;                          // raw data on cpu
#if GPU == 1
    T *data_gpu_;                          // raw data on gpu
#if CUDNN == 1
    cudnnTensorDescriptor_t tensor_desc_;  // tensor descriptor (when this tensor is used as tensor)
    cudnnFilterDescriptor_t filter_desc_;  // filter descriptor (when this tensor is used as filter)
    bool init_tensor_desc_;                // true if tensor descriptor has been initialized
    bool init_filter_desc_;                // true if filter descriptor has been initialized
#endif
#endif
    int n_;                                // batch size
    int c_;                                // channel
    int h_;                                // height
    int w_;                                // width
    int total_size_;                       // total number of elements, n_*c_*h_*w_
    int sample_size_;                      // number of elements in one sample, c_*h_*w_
    int plane_size_;                       // number of elements in one plane, h_*w_
};

#endif
