#pragma once
#include "../utils/helper.cuh"
#include "../utils/helper_host.hpp"

class PoolingLayer{
public:
	void init(cudnnTensorDescriptor_t _x_desc,
						float* _x,
						unsigned x_height,
						unsigned x_width,
						unsigned _window_size,
						unsigned _window_stride,
						unsigned _padding_size,
						unsigned _channel_num,
						float* _p_gradient,
						unsigned _batch_size);

    unsigned y_height;
	unsigned y_width;
    float* y;
    PoolingLayer();
    ~PoolingLayer();

	void forward();
	void backprop();

    cudnnTensorDescriptor_t y_desc;
    cudnnPoolingDescriptor_t pooling_desc;
    float* gradient{nullptr};
    float* p_gradient{nullptr};

private:
    float* x{nullptr};
    cudnnTensorDescriptor_t x_desc;
	unsigned channel_num;
	unsigned batch_size;
	unsigned window_size;
	unsigned window_stride;
	unsigned padding_size;
};