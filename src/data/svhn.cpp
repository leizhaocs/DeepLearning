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
void readSVHNData(const char *filename, Tensor<float> *data)
{
    ifstream file (filename, ios::binary);
    if (file.is_open())
    {
        int magic_number, n, h, w;

        file.read((char*)&magic_number, sizeof(int));
        file.read((char*)&n, sizeof(int));
        file.read((char*)&h, sizeof(int));
        file.read((char*)&w, sizeof(int));

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < h; j++)
            {
                for (int k = 0; k < w; k++)
                {
                    unsigned char temp = 0;
                    file.read((char*)&temp, sizeof(temp));
                    data->data(i, 0, j, k) = temp/255.0f;
                }
            }
        }
    }
}

/* read labels */
void readSVHNLabels(const char *filename, Tensor<float> *labels)
{
    ifstream file (filename, ios::binary);
    if (file.is_open())
    {
        int magic_number, n;

        file.read((char*)&magic_number, sizeof(int));
        file.read((char*)&n, sizeof(int));

        for (int i = 0; i < n; i++)
        {
            unsigned char temp = 0;
            file.read((char*)&temp, sizeof(temp));
            for (int j = 0; j < 10; j++)
            {
                if (j == temp)
                {
                    labels->data(i, j) = 1;
                }
                else
                {
                    labels->data(i, j) = 0;
                }
            }
        }
    }
}

/* load training and test data */
void load_svhn(Tensor<float> **train_data, Tensor<float> **train_labels, Tensor<float> **test_data, Tensor<float> **test_labels,
               int *n_train, int *n_test, int *c, int *h, int *w, int *classes)
{
    *n_train = 73257;
    *n_test = 26032;
    *c = 1;
    *h = 32;
    *w = 32;
    *classes = 10;

    *train_data = new Tensor<float>(*n_train, *c, *h, *w);
    *train_labels = new Tensor<float>(*n_train, 1, 1, *classes);
    *test_data = new Tensor<float>(*n_test, *c, *h, *w);
    *test_labels = new Tensor<float>(*n_test, 1, 1, *classes);

    readSVHNData("data/svhn/train_data", *train_data);
    readSVHNLabels("data/svhn/train_label", *train_labels);
    readSVHNData("data/svhn/test_data", *test_data);
    readSVHNLabels("data/svhn/test_label", *test_labels);

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
