#pragma once
#include "../utils/helper.cuh"
#include "../utils/helper_host.hpp"

class PoolingLayer{
public:
    void init(cudnnTensorDescriptor_t _x_desc,
                        float* _x,
                        unsigned x_height,
                        unsigned x_width,
                        unsigned _pooling_size,
                        unsigned _pooling_stride,
                        unsigned _pooling_type,
                        unsigned _channel_num,
                        unsigned _batch_size);
    unsigned y_height;
    unsigned y_width;
    float* y;
    PoolingLayer();
    ~PoolingLayer();

	void feedForward();
	void backprop();

    cudnnTensorDescriptor_t y_desc;
    cudnnPoolingDescriptor_t pooling_desc;
    float* gradient{nullptr};
    float* last_gradient{nullptr};

private:
    float* x{nullptr};
    cudnnTensorDescriptor_t x_desc;
    unsigned channel_num;
    unsigned batch_size;
    unsigned pooling_size;
    unsigned pooling_stride;
    unsigned pooling_type;
}