#pragma once
#include "../utils/helper.cuh"
#include "../utils/helper_host.hpp"

class PoolingLayer{
public:
    void init(
        unsigned const _pooling_size,
        unsigned const _input_num,
		float* _x,
        float* _p_gradient,
		unsigned const _batch_size
    );

    PoolingLayer();
    ~PoolingLayer();

	void feedForward();
	void backprop();

	unsigned const pooling_size;
    unsigned const batch_size;
    float* y{ nullptr };
	float* gradient{ nullptr };
	float* p_gradient{ nullptr };

private:
	float* x{ nullptr };

	cudnnTensorDescriptor_t y_desc;
    unsigned batch_size;
}