#include "pooling_layer.hpp"

PoolingLayer::PoolingLayer() {}

void PoolingLayer::init(cudnnTensorDescriptor_t _x_desc,
                        float* _x,
                        unsigned x_height,
                        unsigned x_width,
                        unsigned _window_size,
                        unsigned _window_stride,
						unsigned _padding_size,
                        unsigned _channel_num,
						float* _p_gradient,
                        unsigned _batch_size) {
    x_desc = _x_desc;
    x = _x;
	window_size = _window_size;
	window_stride = _window_stride;
	padding_size = _padding_size;
    batch_size = _batch_size;
    channel_num = _channel_num;
    y_height = x_height / _window_stride;
    y_width = x_width / _window_stride;
	p_gradient = _p_gradient;

    checkCUDNN(cudnnCreateTensorDescriptor(&y_desc));
    checkCUDNN(cudnnSetTensor4dDescriptor(y_desc,
		CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size,
        channel_num,
        y_height,
        y_width));

    checkCUDNN(cudnnCreatePoolingDescriptor(&pooling_desc));
    checkCUDNN(cudnnSetPooling2dDescriptor(pooling_desc,
        CUDNN_POOLING_MAX,
        CUDNN_PROPAGATE_NAN,
		window_size, window_size,
		padding_size, padding_size,
		window_stride, window_stride));

	checkCuda(cudaMalloc(
        &gradient, sizeof(float)*batch_size*channel_num*y_height*y_width));
}

PoolingLayer::~PoolingLayer(){
 //   cudnnDestroyTensorDescriptor(x_desc);
	//cudnnDestroyTensorDescriptor(y_desc);
	//cudnnDestroyPoolingDescriptor(pooling_desc);
}

void PoolingLayer::forward() {
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
        y_desc, gradient,
        x_desc, x,
        &beta,
        x_desc,
        p_gradient));
}
