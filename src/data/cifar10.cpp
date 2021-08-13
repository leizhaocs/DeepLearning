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

/* read data */
void readCifar10(const char *filename, Tensor<float> *data, Tensor<float> *labels, int num_images, int offset)
{
    unsigned char temp = 0;

    ifstream file (filename, ios::binary);
    if (file.is_open())
    {
        for (int n = 0; n < num_images; n++)
        {
            file.read((char*)&temp, sizeof(temp));
            for (int l = 0; l < 10; l++)
            {
                if (l == temp)
                {
                    labels->data(n+offset, l) = 1;
                }
                else
                {
                    labels->data(n+offset, l) = 0;
                }
            }

            for (int c = 0; c < 3; c++)
            {
                for (int h = 0; h < 32; h++)
                {
                    for (int w = 0; w < 32; w++)
                    {
                        file.read((char*)&temp, sizeof(temp));
                        data->data(n+offset, c, h, w) = temp/255.0f;
                    }
                }
            }
        }
    }
}

/* load training and test data */
void load_cifar10(Tensor<float> **train_data, Tensor<float> **train_labels, Tensor<float> **test_data, Tensor<float> **test_labels,
                  int *n_train, int *n_test, int *c, int *h, int *w, int *classes)
{
    *n_train = 50000;
    *n_test = 10000;
    *c = 3;
    *h = 32;
    *w = 32;
    *classes = 10;

    *train_data = new Tensor<float>(*n_train, *c, *h, *w);
    *train_labels = new Tensor<float>(*n_train, 1, 1, *classes);
    *test_data = new Tensor<float>(*n_test, *c, *h, *w);
    *test_labels = new Tensor<float>(*n_test, 1, 1, *classes);

    int offset = 0;
    readCifar10("data/cifar10/cifar-10-batches-bin/data_batch_1.bin", *train_data, *train_labels, 10000, offset);
    offset += 10000;
    readCifar10("data/cifar10/cifar-10-batches-bin/data_batch_2.bin", *train_data, *train_labels, 10000, offset);
    offset += 10000;
    readCifar10("data/cifar10/cifar-10-batches-bin/data_batch_3.bin", *train_data, *train_labels, 10000, offset);
    offset += 10000;
    readCifar10("data/cifar10/cifar-10-batches-bin/data_batch_4.bin", *train_data, *train_labels, 10000, offset);
    offset += 10000;
    readCifar10("data/cifar10/cifar-10-batches-bin/data_batch_5.bin", *train_data, *train_labels, 10000, offset);
    readCifar10("data/cifar10/cifar-10-batches-bin/test_batch.bin", *test_data, *test_labels, 10000, 0);

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
