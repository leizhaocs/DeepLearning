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

    /* get data element */
    T &data(int i);
    T &data(int n, int i);
    T &data(int n, int c, int h, int w);

#if GPU == 1
    /* move data from cpu to gpu */
    void toGpu();

    /* move data from gpu to cpu */
    void toCpu();
#endif

    /* get total number of elements */
    int size();

    /* get cpu data pointer */
    T *getCpuPtr();

#if GPU == 1
    /* get cpu data pointer */
    T *getGpuPtr();
#endif

private:
    T *data_cpu_;         // raw data on cpu
#if GPU == 1
    T *data_gpu_;         // raw data on gpu
#endif
    int n_;               // batch size
    int c_;               // channel
    int h_;               // height
    int w_;               // width
    int size_;            // total number of elements
};

#endif
