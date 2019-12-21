#pragma once
#include "../utils/helper_host.h"

class SoftmaxLayer{
public:
	void init(
		unsigned const _class_count,
		unsigned const _input_count,
		float* _x,
		float* _p_gradient,
		unsigned const _batch_size
	);
	SoftmaxLayer();
	~SoftmaxLayer();

	void feedForward();
	void backprop();

	// public members
	unsigned class_count;
	unsigned input_count;
	float* y{ nullptr };
	float* gradient{ nullptr };
	float* p_gradient{ nullptr };
	float* o{ nullptr };
	float* w{ nullptr };

private:
	float* x{ nullptr };
	float* b{ nullptr };
	cudnnTensorDescriptor_t y_desc;
	cudnnTensorDescriptor_t b_desc;
	unsigned batch_size;

	// onevec
	float* onevec{ nullptr };
};