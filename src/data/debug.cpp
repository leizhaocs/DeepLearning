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

/* read training data and labels */
void read(Tensor<float> *data, Tensor<float> *labels)
{
    int n = 1;
    int c = 2;
    int h = 5;
    int w = 5;
    int classes = 2;

    labels->data(0) = 0; labels->data(1) = 1;

    int i = 0;

    data->data(i++) = 0.1; data->data(i++) = -0.2; data->data(i++) = 0.3;  data->data(i++) = -0.4; data->data(i++) = 0.5;
    data->data(i++) = 0.3; data->data(i++) = 0.8;  data->data(i++) = 0.3;  data->data(i++) = 0.4;  data->data(i++) = 0.2;
    data->data(i++) = 0.4; data->data(i++) = 0.7;  data->data(i++) = -0.3; data->data(i++) = 0.2;  data->data(i++) = 0;
    data->data(i++) = 0.2; data->data(i++) = 0.4;  data->data(i++) = 0.6;  data->data(i++) = 0;    data->data(i++) = 0;
    data->data(i++) = 0.4; data->data(i++) = -0.6; data->data(i++) = 0.8;  data->data(i++) = 0;    data->data(i++) = -0.2;

    data->data(i++) = 0.2;  data->data(i++) = 0.2;  data->data(i++) = 0.1;  data->data(i++) = 0.2;  data->data(i++) = 0.7;
    data->data(i++) = -0.6; data->data(i++) = 0.4;  data->data(i++) = 0.2;  data->data(i++) = 0.5;  data->data(i++) = 0.9;
    data->data(i++) = 0.8;  data->data(i++) = -0.6; data->data(i++) = 0.3;  data->data(i++) = 0.3;  data->data(i++) = 0.4;
    data->data(i++) = 0;    data->data(i++) = 0.8;  data->data(i++) = -0.2; data->data(i++) = 0.3;  data->data(i++) = -0.3;
    data->data(i++) = -0.4; data->data(i++) = 0;    data->data(i++) = 0.8;  data->data(i++) = -0.2; data->data(i++) = 0;
}

/* load training and test data */
void load_debug(Tensor<float> **train_data, Tensor<float> **train_labels, Tensor<float> **test_data, Tensor<float> **test_labels,
                int *n_train, int *n_test, int *c, int *h, int *w, int *classes)
{
    *n_train = 1;
    *n_test = 1;
    *c = 2;
    *h = 5;
    *w = 5;
    *classes = 2;

    *train_data = new Tensor<float>(*n_train, *c, *h, *w);
    *train_labels = new Tensor<float>(*n_train, 1, 1, *classes);
    *test_data = new Tensor<float>(*n_test, *c, *h, *w);
    *test_labels = new Tensor<float>(*n_test, 1, 1, *classes);

    read(*train_data, *train_labels);
    read(*test_data, *test_labels);

#if GPU == 1
    if (use_gpu)
    {
        (*train_data)->toGpu();
        (*train_labels)->toGpu();
        (*test_data)->toGpu();
        (*test_labels)->toGpu();
    }
#endif
}
