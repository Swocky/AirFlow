#include "convolution_layer.hpp"

// convlutional layer constructor
ConvolutionalLayer::ConvolutionalLayer(
	unsigned const _kernel_count, 
	unsigned const _kernel_size,
	unsigned const _stride,
	unsigned const _channel_count,
	unsigned const _input_x,
	unsigned const _input_y,
	float* _x,
	float* _p_gradient,
	unsigned const _batch_size
) :
	kernel_count(_kernel_count),
	kernel_size(_kernel_size),
	stride(_stride),
	channel_count(_channel_count),
	input_x(_input_x),
	input_y(_input_y),
	x(_x),
	p_gradient(_p_gradient),
	output_x((_input_x - _kernel_size) / _stride + 1),
	output_y((_input_y - _kernel_size) / _stride + 1),
	batch_size(_batch_size){
	// creating & setting data descriptor 'x_desc'
	checkCUDNN(cudnnCreateTensorDescriptor(&x_desc));
	checkCUDNN(cudnnSetTensor4dDescriptor(
		x_desc,
		CUDNN_TENSOR_NCHW,
		CUDNN_DATA_FLOAT,
		batch_size,
		channel_count,
		input_y,
		input_x
	));


	// allocating weights 'w'
	size_t const w_bytes = kernel_count * pow(kernel_size, 2) * channel_count * sizeof(float);
	checkCuda(cudaMalloc(&w, w_bytes));
	// randomizing weights 'w'
	Randomize(w, w_bytes / sizeof(float), 0.03f);
	// creating & setting filter descriptions for weights 'w_desc'
	checkCUDNN(cudnnCreateFilterDescriptor(&w_desc));
	checkCUDNN(cudnnSetFilter4dDescriptor(
		w_desc,
		CUDNN_DATA_FLOAT,
		CUDNN_TENSOR_NCHW,
		kernel_count,
		channel_count,
		kernel_size, kernel_size
	));


	// allocating biases 'b'
	size_t const b_bytes = kernel_count * sizeof(float);
	checkCuda(cudaMalloc(&b, b_bytes), 0.01f);
	// randomizing biases 'b'
	Randomize(b, b_bytes / sizeof(float));
	// creating & setting tensor descriptions for biases 'b_desc'
	checkCUDNN(cudnnCreateTensorDescriptor(&b_desc));
	checkCUDNN(cudnnSetTensor4dDescriptor(
		b_desc,
		CUDNN_TENSOR_NCHW,
		CUDNN_DATA_FLOAT,
		1, kernel_count, 1, 1
	));


	// allocating for output before the relu 'o' & after relu 'y'
	size_t const y_bytes = batch_size * kernel_count * outputY * outputX * sizeof(float);
	checkCuda(cudaMalloc(&o, y_bytes));
	checkCuda(cudaMalloc(&y, y_bytes));
	// creating & setting tensor descriptions for 'o' & 'y'
	checkCUDNN(cudnnCreateTensorDescriptor(&y_desc));
	checkCUDNN(cudnnSetTensor4dDescriptor(
		y_desc,
		CUDNN_TENSOR_NCHW,
		CUDNN_DATA_FLOAT,
		batch_size,
		kernel_count,
		outputY, outputX
	));


	// allocating gradients 'gradient', it's size is the same y, and when gradient description is required, y_desc can be used
	size_t const gradient_bytes = batch_size * kernel_count * outputY * outputX * sizeof(float);
	checkCuda(cudaMalloc(&gradient, gradient_bytes));

	// creating & setting activation for the layers, relu is used
	checkCUDNN(cudnnCreateActivationDescriptor(&activationDesc));
	checkCUDNN(cudnnSetActivationDescriptor(
		activationDesc,
		CUDNN_ACTIVATION_ELU,
		CUDNN_NOT_PROPAGATE_NAN,
		1.0
	));


	// creating & setting convolution descriptor 'conv_desc'
	checkCUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
	checkCUDNN(cudnnSetConvolution2dDescriptor(
		conv_desc,
		0, 0, /*zero padding*/
		stride, stride,
		1, 1, /*normal dilation*/
		CUDNN_CROSS_CORRELATION,
		CUDNN_DATA_FLOAT
	));


	// getting convolution forward algorithm 'fwd_algo'
	checkCUDNN(cudnnGetConvolutionForwardAlgorithm(
		cudnn,
		x_desc,
		w_desc,
		conv_desc,
		y_desc,
		CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
		0, /*unlimited for now, later will get needed workspace*/
		&fwd_algo
	));

	// getting forward workspace size & allocating 'forward_workspace'
	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
		cudnn,
		x_desc,
		w_desc,
		conv_desc,
		y_desc,
		fwd_algo,
		&forward_workspace_bytes
	));
	checkCuda(cudaMalloc(&forward_workspace, forward_workspace_bytes));


	// getting convolution backward filter algorithm 'bwd_filter_algo'
	checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(
		cudnn,
		x_desc,
		y_desc, /*same dimensions as dy_desc*/
		conv_desc,
		w_desc, /*same dimensions as dw_desc*/
		CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
		0, /*unlimited for now, later will get needed workspace*/
		&bwd_filter_algo
	));
	// getting backward filter workspace size & allocating 'backward_filter_workspace'
	checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
		cudnn,
		x_desc,
		y_desc,
		conv_desc,
		w_desc,
		bwd_filter_algo,
		&backward_filter_workspace_bytes
	));
	cudaDeviceSynchronize();
	//std::cout << "trying to allocate: " << backward_filter_workspace_bytes << " bytes.\n";
	checkCuda(cudaMalloc(&backward_filter_workspace, backward_filter_workspace_bytes));


	// getting convolution backward data algorithm 'bwd_data_algo'
	checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(
		cudnn,
		w_desc,
		y_desc,
		conv_desc,
		x_desc,
		CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
		0,
		&bwd_data_algo
	));


	// getting backward data workspace size & allocating 'backward_data_workspace'
	checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
		cudnn,
		w_desc,
		y_desc,
		conv_desc,
		x_desc,
		bwd_data_algo,
		&backward_data_workspace_bytes
	));
	checkCuda(cudaMalloc(&backward_data_workspace, backward_data_workspace_bytes));
}


ConvolutionalLayer::~ConvolutionalLayer(){
}

void ConvolutionalLayer::setX(float* new_x) { x = new_x; }

void ConvolutionalLayer::convolutionForward()
{
	const float alpha = 1.0f, beta = 0.0f;

	// convolution, o = x*w
	checkCUDNN(cudnnConvolutionForward(
		cudnn,
		&alpha,
		x_desc, x,
		w_desc, w,
		conv_desc,
		fwd_algo,
		forward_workspace, forward_workspace_bytes,
		&beta,
		y_desc, o
	));

	// adding bias, o += b
	checkCUDNN(cudnnAddTensor(
		cudnn,
		&alpha,
		b_desc, b,
		&alpha,
		y_desc, o
	));

	// relu, y = RELU(o), 'RELU()' is a symbol not an actual function
	checkCUDNN(cudnnActivationForward(
		cudnn,
		activationDesc,
		&alpha,
		y_desc, o,
		&beta,
		y_desc, y
	));


}

void ConvolutionalLayer::convolutionBackward()
{
	// used SGD momentum to update weights
	// if wanted to use only sgd, change momentum_b with beta, momentum_g with alpha 
	// in 'cudnnConvolutionBackwardFilter' and 'cudnnConvolutionBackwardBias'

	const float alpha = 1.0f, beta = 0.0f, momentum_b = 0.9f;
	const float momentum_g = 1.0f - momentum_b;
	//const float p_gradientMul = 1.5f;

	// not sure if it works...
	checkCUDNN(cudnnActivationBackward(
		cudnn,
		activationDesc,
		&alpha,
		y_desc, y,
		y_desc, gradient,
		y_desc, o,
		&beta,
		y_desc, gradient
	));

	// updating weights
	checkCUDNN(cudnnConvolutionBackwardFilter(
		cudnn,
		&learning_rate,
		x_desc, x,
		y_desc, gradient,
		conv_desc,
		bwd_filter_algo,
		backward_filter_workspace,
		backward_filter_workspace_bytes,
		&alpha,
		w_desc, w
	));

	// updating biases
	checkCUDNN(cudnnConvolutionBackwardBias(
		cudnn,
		&learning_rate,
		y_desc, gradient,
		&alpha,
		b_desc, b
	));

	// passing the gradient to backward layers
	checkCUDNN(cudnnConvolutionBackwardData(
		cudnn,
		&momentum_g,
		w_desc, w,
		y_desc, gradient,
		conv_desc,
		bwd_data_algo,
		backward_data_workspace,
		backward_data_workspace_bytes,
		&momentum_b,
		x_desc, p_gradient
	));
}
