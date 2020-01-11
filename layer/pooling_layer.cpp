#include "pooling_layer.hpp"

PoolingLayer::PoolingLayer() {}

void PoolingLayer::init(cudnnTensorDescriptor_t _x_desc,   // 前一层的输入描述符
                        float* _x,                         // 输入数据
                        unsigned x_height,                 // feature map长度
                        unsigned x_width,                  // feature map宽度
                        unsigned _window_size,             // 池化层大小
                        unsigned _window_stride,           // 池化层步幅
						unsigned _padding_size,            // 池化层用于保持边界信息的拓展边缘大小
                        unsigned _channel_num,             // 输入通道数（与输出通道数相同）
						float* _p_gradient,                // 前一层梯度
                        unsigned _batch_size) {            // batch大小
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
	checkCuda(cudaMalloc(
		&y, sizeof(float)*batch_size*channel_num*y_height*y_width));
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
