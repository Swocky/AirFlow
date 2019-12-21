#pragma once
#include "../utils/helper.cuh"
#include "../utils/helper_host.hpp"

class ConvolutionalLayer{
public:
	ConvolutionalLayer(
		unsigned const _kernel_num,
		unsigned const _kernel_size,
		unsigned const _stride,
		unsigned const _channel_num,
		unsigned const _input_x,
		unsigned const _input_y,
		float* x,
		float* _p_gradient,
		unsigned const _batch_size
	);
	~ConvolutionalLayer();

	void setX(float* new_x);

	void convolutionForward();
	void convolutionBackward();

	// understood varibles
	const unsigned output_x, output_y;

	float* y{ nullptr }; // output after relu, y = RELU(o); for relu cudnnActivationForward is used
	float* gradient{ nullptr }; // gradient is the error of every 'y'. So it's size is the same as 'y' or 'o'.
	float* p_gradient{ nullptr };
	float* x{ nullptr }; // pointer to data that the layer will convolute
	float* w{ nullptr }; // weights of the layer
	float* b{ nullptr }; // biases of the layer
	float* o{ nullptr }; // output before the relu, o = x*w+b


private:
	// defining vars of the convolution layer
	// there is no padding for now
	unsigned const kernel_num;
	unsigned const kernel_size;
	unsigned const stride;
	unsigned const channel_num;
	unsigned const input_x, input_y;
	unsigned const batch_size;



	cudnnConvolutionFwdAlgo_t fwd_algo; // algorithm for forward propagation, used in the cudnnConvolutionForward func.
	cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo;	// algorithm for updating filter,
														// used in the cudnnConvolutionBackwardFilter func.
	cudnnConvolutionBwdDataAlgo_t bwd_data_algo;

	cudnnActivationDescriptor_t activationDesc;

	size_t forward_workspace_bytes; // size in bytes of forward_workspace
	void* forward_workspace{ nullptr }; // needed for cudnnConvolutionForward func.

	size_t backward_filter_workspace_bytes; // size in bytes of backward_filter_workspace
	void* backward_filter_workspace{ nullptr }; // needed for cudnnConvolutionBackwardFilter and...........

	size_t backward_data_workspace_bytes; // size in bytes of backward_data_workspace
	void* backward_data_workspace{ nullptr }; // needed for cudnnConvolutionDataFilter and...........

	// Tensor descriptions for input data, weights, biases, convolution and output respectively
	cudnnTensorDescriptor_t x_desc;
	cudnnFilterDescriptor_t w_desc;
	cudnnTensorDescriptor_t b_desc;
	cudnnConvolutionDescriptor_t conv_desc;
	cudnnTensorDescriptor_t y_desc;
};

