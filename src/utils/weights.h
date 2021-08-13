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

#ifndef _WEIGHTS_H_
#define _WEIGHTS_H_

#include "includes.h"

/* store weights file */
void store_weights(const char *weights_file, float *weights, int num_weights);

/* load weights file */
void load_weights(const char *weights_file, float *weights, int num_weights);

/* randomly initializing weights */
void random_weights(float *weights, int num_weights, float mean=0, float stddev=1, float scale=0.1);

#endif
