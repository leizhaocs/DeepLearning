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

#ifndef _OBSERVATION_H_
#define _OBSERVATION_H_

#include "includes.h"

/* one observation, a sequence of grey scale images */
class Observation
{
public:
    /* constructor */
    Observation(int n, int h, int w);

    /* destructor */
    ~Observation();

    /* copy into the nth image in images_ */
    void setImage(int n, float *p);

    /* get images pointer */
    Tensor<float> *getImagesPtr();

    /* get length of the sequence */
    int getLength();

    /* get image height */
    int getHeight();

    /* get image width */
    int getWidth();

private:
    int n_;                  // length of the sequence
    int h_;                  // image height
    int w_;                  // image width

    Tensor<float> *images_;  // images
};

#endif
