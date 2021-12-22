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

#ifndef _LAYER_CONV_H__
#define _LAYER_CONV_H__

#include "includes.h"

/* convolution layer */
class LayerConv : public Layer
{
public:
    /* constructor */
    LayerConv(Params *params, Layer *prev_layer);

    /* destructor */
    ~LayerConv();

    /* forward propagation */
    void cpu_forward(int realBatchSize, bool train);

    /* backward propagation */
    void cpu_backward(int realBatchSize);

    /* update weights and biases */
    void cpu_update(int realBatchSize, float lr);

#if GPU == 1
    /* forward propagation */
    void gpu_forward(int realBatchSize, bool train);

    /* backward propagation */
    void gpu_backward(int realBatchSize);

    /* update weights and biases */
    void gpu_update(int realBatchSize, float lr);
#endif

    /* initialize weights */
    void initWeights(float *weights, int &offset);

    /* get weights */
    void getWeights(float *weights, int &offset);

    /* get number of weights in this layer */
    vector<int> getNumWeights();

private:
    int num_filters_;                                       // number of filters
    int filter_h_;                                          // filter height
    int filter_w_;                                          // filter width

    int stride_h_;                                          // stride height
    int stride_w_;                                          // stride width

    int padding_h_;                                         // padding height
    int padding_w_;                                         // padding width

    Tensor<float> *filters_;                                // filters (f,c,h,w)
    Tensor<float> *biases_;                                 // biases (f)

    Tensor<float> *grad_filters_;                           // gradients of filters (f,c,h,w)
    Tensor<float> *grad_biases_;                            // gradients of biases (f)

#if GPU == 1
#if CUDNN == 1
    cudnnConvolutionDescriptor_t conv_desc_;                // convolution descriptor
    cudnnConvolutionFwdAlgo_t conv_fwd_algo_;               // forward propagation algorithm
    cudnnConvolutionBwdDataAlgo_t conv_bwd_data_algo_;      // data backward propagation algorithm
    cudnnConvolutionBwdFilterAlgo_t conv_bwd_filter_algo_;  // filter backward propagation algorithm

    size_t workspace_size_;                                 // size of the workspace
    void **workspace_;                                      // workspace for convolution
#endif
#endif
};

#endif
