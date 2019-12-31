#include "pooling_layer.hpp"

PoolingLayer::PoolingLayer() {}

void PoolingLayer::init(cudnnTensorDescriptor_t _x_desc,
                        float* _x,
                        unsigned x_height,
                        unsigned x_width,
                        unsigned _pooling_size,
                        unsigned _pooling_stride,
                        unsigned _pooling_type,
                        unsigned _channel_num,
                        unsigned _batch_size,
                        float* _last_gradient) {
    x_desc = _x_desc;
    x = _x;
    pooling_size = _pooling_size;
    pooling_stride = _pooling_stride;
    batch_size = _batch_size;
    channel_num = _channel_num;
    y_height = x_height / pooling_stride;
    y_width = x_width / pooling_stride;
    pooling_type = _pooling_type;
    last_gradient = _last_gradient;

    checkCUDNN(cudnnCreateTensorDescriptor(&y_desc));
    checkCUDNN(cudnnSetTensor4dDescriptor(y_desc,
        CUDNN_TENSOR_NHWC,
        CUDNN_DATA_FLOAT,
        batch_size,
        channel_num,
        y_height,
        y_width));

    checkCUDNN(cudnnCreatePoolingDescriptor(&pooling_desc));
    checkCUDNN(cudnnSetPooling2dDescriptor(pooling_descriptor,
        pooling_type,//CUDNN_POOLING_MAX
        CUDNN_PROPAGATE_NAN,
        pooling_size, pooling_size,
        0,0,
        pooling_stride, pooling_stride));

    checkCudaErrors(cudaMalloc(
        &dpool, sizeof(float)*batch_size*channel_num*x_height*x_width));
}

void PoolingLayer::~PoolingLayer(){
    cudnnDestroyTensorDescriptor(x_desc);
	cudnnDestroyTensorDescriptor(y_desc);
	cudnnDestroyPoolingDescriptor(pooling_desc);
}

void PoolingLayer::feedForward() {
    const float alpha = 1.0f, beta = 0.0f;
    cudnnPoolingForward(
        cudnn,
        pooling_desc,
        &alpha,
        x_desc,
        x,
        &beta,
        y_desc,
        y);
}

void PoolingLayer::backprop() {
    const float alpha = 1.0f, beta = 0.0f;
    checkCUDNN(cudnnPoolingBackward(
        cudnn,
        pooling_desc,
        &alpha,
        y_desc, y,
        y_desc, next_gradient,
        x_desc, x,
        &beta,
        x_desc,
        gradient));
}
