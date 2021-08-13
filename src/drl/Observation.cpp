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
Observation::Observation(int n, int h, int w)
{
    n_ = n;
    h_ = h;
    w_ = w;

    images_ = new Tensor<float>(1, n_, h_, w_);
}

/* destructor */
Observation::~Observation()
{
    delete images_;
}

/* copy into the nth image in images_ */
void Observation::setImage(int n, float *p)
{
    for (int h = 0; h < h_; h++)
    {
        for (int w = 0; w < w_; w++)
        {
            images_->data(0, n, h, w) = p[h*w_+w];
        }
    }

#if GPU == 1
    if (use_gpu)
    {
        images_->toGpu();
    }
#endif
}

/* get images pointer */
Tensor<float> *Observation::getImagesPtr()
{
    return images_;
}

/* get length of the sequence */
int Observation::getLength()
{
    return n_;
}

/* get image height */
int Observation::getHeight()
{
    return h_;
}

/* get image width */
int Observation::getWidth()
{
    return w_;
}
