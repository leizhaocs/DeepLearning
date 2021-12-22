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

/* implemented in cpp files in data folder */
void load_mnist(Tensor<float> **train_data, Tensor<float> **train_labels, Tensor<float> **test_data, Tensor<float> **test_labels,
                int *n_train, int *n_test, int *c, int *h, int *w, int *classes);
void load_cifar10(Tensor<float> **train_data, Tensor<float> **train_labels, Tensor<float> **test_data, Tensor<float> **test_labels,
                  int *n_train, int *n_test, int *c, int *h, int *w, int *classes);
void load_svhn(Tensor<float> **train_data, Tensor<float> **train_labels, Tensor<float> **test_data, Tensor<float> **test_labels,
               int *n_train, int *n_test, int *c, int *h, int *w, int *classes);
void load_debug(Tensor<float> **train_data, Tensor<float> **train_labels, Tensor<float> **test_data, Tensor<float> **test_labels,
                int *n_train, int *n_test, int *c, int *h, int *w, int *classes);

/* caculate the prediction accuracy */
float accuracy(Tensor<float> *outputs, Tensor<float> *ground_truth, int num_samples, int num_classes)
{
#if GPU == 1
    if (use_gpu)
    {
        outputs->toCpu();
    }
#endif

    int correct = 0, wrong = 0;
    for (int i = 0; i < num_samples; i++)
    {
        float pred_max = 0, max = 0;
        int pred_digit = -1, digit = -2;
        for (int j = 0; j < num_classes; j++)
        {
            if (outputs->data(i,j) > pred_max)
            {
                pred_max = outputs->data(i,j);
                pred_digit = j;
            }
            if (ground_truth->data(i,j) > max)
            {
                max = ground_truth->data(i,j);
                digit = j;
            }
        }
        if (digit == pred_digit)
        {
            correct++;
        }
        else
        {
            wrong++;
        }
    }

    return ((float)correct)/num_samples;
}

/* inference the entire test data set, print accuracy */
void classify(Net *net, Tensor<float> *test_data, Tensor<float> *test_labels, Tensor<float> *test_pred,
              int n_test, int c, int h, int w, int classes, int batchSize)
{
    auto start = chrono::high_resolution_clock::now();

    int numbatches = DIVUP(n_test, batchSize);

    for (int i = 0, offset = 0; i < numbatches; i++)
    {
        int realBatchSize = min(n_test-offset, batchSize);

        net->initForward(test_data, offset, 0, realBatchSize, c, h, w);
        net->forward(realBatchSize, false);

        net->getOutputs(test_pred, offset, 0, realBatchSize, classes);

        offset += realBatchSize;
    }

    float test_acc = accuracy(test_pred, test_labels, n_test, classes);

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> diff = end-start;

    printf("test acc: %f    test time: %fs\n", test_acc, diff.count());
}

/* train the entire training data set for a number of epochs */
void train(Net *net,
           Tensor<float> *train_data, Tensor<float> *test_data,
           Tensor<float> *train_labels, Tensor<float> *test_labels,
           Tensor<float> *train_pred, Tensor<float> *test_pred,
           int n_train, int n_test, int c, int h, int w, int classes,
           int batchSize, float lr_begin, float lr_decay, int num_epochs, int show_acc)
{
    int numbatches = DIVUP(n_train, batchSize);
    float lr = lr_begin;

    auto start = chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < num_epochs; epoch++)
    {
        printf("Epoch: %d/%d    lr:%f    ", epoch+1, num_epochs, lr);

        auto epoch_start = chrono::high_resolution_clock::now();

        for (int i = 0, offset = 0; i < numbatches; i++)
        {
            int realBatchSize = min(n_train-offset, batchSize);

            net->initForward(train_data, offset, 0, realBatchSize, c, h, w);
            net->forward(realBatchSize, true);

            net->getOutputs(train_pred, offset, 0, realBatchSize, classes);
            //net->loss(train_labels, offset, realBatchSize, classes);

            net->initBackward(train_labels, offset, realBatchSize, classes);
            net->backward(realBatchSize);
            net->update(realBatchSize, lr);

            offset += realBatchSize;
        }

        auto epoch_end = chrono::high_resolution_clock::now();
        chrono::duration<double> epoch_diff = epoch_end-epoch_start;

        lr -= lr_decay;

        if (show_acc == 0)
        {
            printf("time: %fs\n", epoch_diff.count());
        }
        else if (show_acc == 1)
        {
            float train_acc = accuracy(train_pred, train_labels, n_train, classes);
            printf("training acc: %f    traing time: %fs\n", train_acc, epoch_diff.count());
        }
        else if (show_acc == 2)
        {
            float train_acc = accuracy(train_pred, train_labels, n_train, classes);
            printf("training acc: %f    training time: %fs    ", train_acc, epoch_diff.count());
            classify(net, test_data, test_labels, test_pred, n_test, c, h, w, classes, batchSize);
        }
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> diff = end-start;

    printf("================================\n");
    printf("        Training Results        \n");
    printf("time: %fs\n", diff.count());
}

/* main function */
int main_dnn(int argc, char *argv[])
{
    /* network parameters */
    Net *net;                     // network
    Params *layer_params;         // config file parameters
    int num_layers;               // number of layers
    int num_epochs;               // number of epochs
    int batchSize;                // batch size
    float lr_begin;               // learning rate
    float lr_decay;               // decay of learning rate
    string loss;                  // loss function
    int show_acc;                 // show accuracy in each epoch
    float *weights;               // weights of network

    /* datasets */
    int n_train = 0;              // number of training samples
    int n_test = 0;               // number of test samples
    int c = 0;                    // number of channels of input image
    int h = 0;                    // input image height
    int w = 0;                    // input image width
    int classes = 0;              // number of classes
    Tensor<float> *train_data;    // training data
    Tensor<float> *train_labels;  // training labels
    Tensor<float> *train_pred;    // predictions on training data
    Tensor<float> *test_data;     // test data
    Tensor<float> *test_labels;   // test labels
    Tensor<float> *test_pred;     // predictions on test data

    cout<<"Loading data..."<<endl;

    if (strcmp(argv[3], "mnist") == 0)
    {
        load_mnist(&train_data, &train_labels, &test_data, &test_labels, &n_train, &n_test, &c, &h, &w, &classes);
    }
    else if (strcmp(argv[3], "cifar10") == 0)
    {
        load_cifar10(&train_data, &train_labels, &test_data, &test_labels, &n_train, &n_test, &c, &h, &w, &classes);
    }
    else if (strcmp(argv[3], "svhn") == 0)
    {
        load_svhn(&train_data, &train_labels, &test_data, &test_labels, &n_train, &n_test, &c, &h, &w, &classes);
    }
    else if (strcmp(argv[3], "debug") == 0)
    {
        load_debug(&train_data, &train_labels, &test_data, &test_labels, &n_train, &n_test, &c, &h, &w, &classes);
    }
    else
    {
        Assert(false, "Unsupported dataset.");
    }

    train_pred = new Tensor<float>(n_train, 1, 1, classes);
    test_pred = new Tensor<float>(n_test, 1, 1, classes);

    cout<<"Building network..."<<endl;

    layer_params = build_network(argv[4], num_layers, num_epochs, batchSize, lr_begin, lr_decay, loss, show_acc);
    net = new Net(layer_params, num_layers, batchSize, loss);
    int num_weights = net->getNumWeights(NULL);
    weights = new float[num_weights];

    if (strcmp(argv[2], "train") == 0)
    {
        if (strcmp(argv[5], "null") == 0)
        {
            cout<<"Initializing weights..."<<endl;

            net->initWeights(NULL);
        }
        else
        {
            cout<<"Loading weights..."<<endl;

            load_weights(argv[5], weights, num_weights);
            net->initWeights(weights);
        }

        MemoryMonitor::instance()->printCpuMemory();
#if GPU == 1
        MemoryMonitor::instance()->printGpuMemory();
#endif

        cout<<"Training..."<<endl;

        train(net, train_data, test_data, train_labels, test_labels, train_pred, test_pred,
              n_train, n_test, c, h, w, classes,
              batchSize, lr_begin, lr_decay, num_epochs, show_acc);

        if (strcmp(argv[6], "null") != 0)
        {
            cout<<"Saving weights..."<<endl;

            net->getWeights(weights);
            store_weights(argv[6], weights, num_weights);
        }
    }
    else if (strcmp(argv[2], "test") == 0)
    {
        cout<<"Loading weights..."<<endl;

        load_weights(argv[5], weights, num_weights);
        net->initWeights(weights);

        MemoryMonitor::instance()->printCpuMemory();
#if GPU == 1
        MemoryMonitor::instance()->printGpuMemory();
#endif

        cout<<"Classifying..."<<endl;

        classify(net, test_data, test_labels, test_pred, n_test, c, h, w, classes, batchSize);
    }
    else
    {
        Assert(false, "Neither train nor test");
    }

    delete net;
    delete [] layer_params;
    delete [] weights;
    delete train_data;
    delete train_labels;
    delete train_pred;
    delete test_data;
    delete test_labels;
    delete test_pred;

    return 0;
}
