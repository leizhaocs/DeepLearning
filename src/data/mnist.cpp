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

/* change from little endian to big endian */
int reverseInt (int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 0xff;
    ch2 = (i>>8) & 0xff;
    ch3 = (i>>16) & 0xff;
    ch4 = (i>>24) & 0xff;
    return ((int)ch1<<24) + ((int)ch2<<16) + ((int)ch3<<8) + ch4;
}

/* read data */
void readMNISTData(const char *filename, Tensor<float> *data)
{
    ifstream file (filename, ios::binary);
    if (file.is_open())
    {
        int magic_number, n, h, w;

        file.read((char*)&magic_number, sizeof(int));
        magic_number = reverseInt(magic_number);
        file.read((char*)&n, sizeof(int));
        n = reverseInt(n);
        file.read((char*)&h, sizeof(int));
        h = reverseInt(h);
        file.read((char*)&w, sizeof(int));
        w = reverseInt(w);

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
void readMNISTLabels(const char *filename, Tensor<float> *labels)
{
    ifstream file (filename, ios::binary);
    if (file.is_open())
    {
        int magic_number, n;

        file.read((char*)&magic_number, sizeof(int));
        magic_number = reverseInt(magic_number);
        file.read((char*)&n, sizeof(int));
        n = reverseInt(n);

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
void load_mnist(Tensor<float> **train_data, Tensor<float> **train_labels, Tensor<float> **test_data, Tensor<float> **test_labels,
                int *n_train, int *n_test, int *c, int *h, int *w, int *classes)
{
    *n_train = 60000;
    *n_test = 10000;
    *c = 1;
    *h = 28;
    *w = 28;
    *classes = 10;

    *train_data = new Tensor<float>(*n_train, *c, *h, *w);
    *train_labels = new Tensor<float>(*n_train, 1, 1, *classes);
    *test_data = new Tensor<float>(*n_test, *c, *h, *w);
    *test_labels = new Tensor<float>(*n_test, 1, 1, *classes);

    readMNISTData("data/mnist/train-images-idx3-ubyte", *train_data);
    readMNISTLabels("data/mnist/train-labels-idx1-ubyte", *train_labels);
    readMNISTData("data/mnist/t10k-images-idx3-ubyte", *test_data);
    readMNISTLabels("data/mnist/t10k-labels-idx1-ubyte", *test_labels);

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
