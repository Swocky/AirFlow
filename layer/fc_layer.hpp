#pragma once
#include "../utils/helper_host.hpp"

class FullyConnectedLayer
{
public:
	FullyConnectedLayer(
		
	);
	~FullyConnectedLayer();

	void init(
		unsigned const _neuron_num,
		unsigned const _input_num,
		float* _x,
		float* _p_gradient,
		unsigned const _batch_size
	);

	void feedForward();
	void backprop();


	// public members
	unsigned neuron_num; //
	unsigned input_num; //
	float* y{ nullptr }; //
	float* gradient{ nullptr }; //
	float* p_gradient{ nullptr }; // previous layer's gradient
	float* w{ nullptr }; //

private:
	float* x{ nullptr }; //
	float* b{ nullptr }; //
	float* o{ nullptr }; //
	cudnnTensorDescriptor_t y_desc;
	cudnnTensorDescriptor_t b_desc;
	unsigned batch_size;

	// onevec
	float* onevec{ nullptr };

	cudnnActivationDescriptor_t activation_desc; // relu
};

