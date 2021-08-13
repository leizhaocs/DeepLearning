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

#ifndef _GYM_H_
#define _GYM_H_

#include "includes.h"

/* send all data */
void sendall(int sock, void *data, int size);

/* recieve all data */
void recvall(int sock, void *data, int size);

/* connect to gym server */
int gym_connect();

/* disconnect from gym server */
void gym_disconnect();

/* create a gym environment */
void gym_make(string env_name);

/* close a gym environment */
void gym_close();

/* reset a gym environment, and get the observation size */
void gym_reset();

/* get the dimension of image */
void gym_getImageDimension(int *channels, int *height, int *width);

/* render the gym */
void gym_render(float *image, int size);

/* run one step on the environment, return finished(1) or not(0) */
float gym_step_discrete(int action, int *done);

/* run one step on the environment, return finished(1) or not(0) */
float gym_step_box(int n, float *action, int *done);

#endif
