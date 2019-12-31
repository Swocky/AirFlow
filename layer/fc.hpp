#pragma once
#include "../utils/helper_host.hpp"
#include "fc_layer.hpp"

class FullyConnected{
public:
	FullyConnected(); // default constructor
	~FullyConnected();

	void init(
		const std::vector<unsigned> _neuron_nums,
		const unsigned _input_num,
		float* _x,
		float* _p_gradient,
		const unsigned _batch_size
	);

	void forward();
	void backprop();
	
	// public members
	float* o{ nullptr };
	float* y{ nullptr }; // output of the last fullyConnectedLayer
	float* gradient{ nullptr }; // last layers gradient
	float* p_gradient{ nullptr };

private:
	// defining vars of the fully connected
	std::vector<unsigned> neuron_nums;
	unsigned input_num = 0;
	float* x{ nullptr }; // output of the convolution layers, device ptr
	unsigned batch_size = 0;

	std::vector<FullyConnectedLayer> fc_layers;
};
