#include "fc_layer.hpp"

void FullyConnectedLayer::init(
	unsigned const _neuron_num,
	unsigned const _input_num,
	float* _x,
	float* _p_gradient,
	unsigned const _batch_size
){
	neuron_num = _neuron_num;
	input_num = _input_num;
	x =_x;
	p_gradient = _p_gradient;
	batch_size =_batch_size;
	 std::cout << "input num: " << input_num << '\n';
	 std::cout << "neuron num: " << neuron_num << '\n';
	// std::cout << "batch_size: " << batch_size << '\n';
	// allocating weights, 'w'
	size_t const w_bytes = input_num * neuron_num * sizeof(float);
	//std::cout << "Allocating: " << w_bytes / 1024.0f / 1024.0f << "Mb.\n";
	checkCuda(cudaMalloc(&w, w_bytes));
	randomize(w, w_bytes / sizeof(float), 0.03f);
	// allocating biases, 'b' & creating and setting b_desc
	size_t const b_bytes = neuron_num * sizeof(float);
	checkCuda(cudaMalloc(&b, b_bytes));
	randomize(b, neuron_num, 0.01f);
	checkCUDNN(cudnnCreateTensorDescriptor(&b_desc));
	checkCUDNN(cudnnSetTensor4dDescriptor(
		b_desc,
		CUDNN_TENSOR_NCHW,
		CUDNN_DATA_FLOAT,
		1,
		neuron_num,
		1, 1
	));


	// allocating output before relu, 'o' & relu, 'y' & creating and setting y_desc
	size_t const y_bytes = neuron_num * batch_size * sizeof(float);
	checkCuda(cudaMalloc(&o, y_bytes));
	checkCuda(cudaMalloc(&y, y_bytes));
	checkCUDNN(cudnnCreateTensorDescriptor(&y_desc));
	checkCUDNN(cudnnSetTensor4dDescriptor(
		y_desc,
		CUDNN_TENSOR_NCHW,
		CUDNN_DATA_FLOAT,
		batch_size,
		neuron_num,
		1, 1
	));

	// creating & setting activation for the layers, relu is used
	checkCUDNN(cudnnCreateActivationDescriptor(&activation_desc));
	checkCUDNN(cudnnSetActivationDescriptor(
		activation_desc,
		CUDNN_ACTIVATION_RELU,
		CUDNN_NOT_PROPAGATE_NAN,
		1.0
	));

	// allocating gradient for this layers output, 'gradient'
	checkCuda(cudaMalloc(&gradient, y_bytes));


	// allocating onevec
	checkCuda(cudaMalloc(&onevec, batch_size * sizeof(float)));
	std::vector<float> temp_onevec(batch_size, 1.0f);
	checkCuda(cudaMemcpy(onevec, temp_onevec.data(), batch_size * sizeof(float), cudaMemcpyHostToDevice));
}

FullyConnectedLayer::FullyConnectedLayer(){
}

FullyConnectedLayer::~FullyConnectedLayer(){
}

void FullyConnectedLayer::forward(){
	const float alpha = 1.0f, beta = 0.0f;
	// multiplying by weights, o = x*w;
	cublasSgemm_v2(
		cublas, CUBLAS_OP_N, CUBLAS_OP_N,
		neuron_num, batch_size, input_num,
		&alpha,
		w, neuron_num,
		x, input_num,
		&beta,
		o, neuron_num
	);

	// adding biases, o += b;
	checkCUDNN(cudnnAddTensor(
		cudnn,
		&alpha,
		b_desc, b,
		&alpha,
		y_desc, o
	));

	// relu
	checkCUDNN(cudnnActivationForward(
		cudnn,
		activation_desc,
		&alpha,
		y_desc, o,
		&beta,
		y_desc, y
	));
}

void FullyConnectedLayer::backprop()
{
	const float alpha = 1.0f, beta = 0.0f;

	// taking derivative of activation func.
	checkCUDNN(cudnnActivationBackward(
		cudnn,
		activation_desc,
		&alpha,
		y_desc, y,
		y_desc, gradient,
		y_desc, o,
		&beta,
		y_desc, gradient
	));

	// passing gradient to previous layer
	cublasSgemm_v2(
		cublas, CUBLAS_OP_T, CUBLAS_OP_N,
		input_num, batch_size, neuron_num,
		&alpha,
		w, neuron_num,
		gradient, neuron_num,
		&beta,
		p_gradient, input_num
	);
	// updating weights, 'w'
	cublasSgemm_v2(
		cublas, CUBLAS_OP_N, CUBLAS_OP_T,
		neuron_num, input_num, batch_size,
		&learning_rate,
		gradient, neuron_num,
		x, input_num,
		&alpha,
		w, neuron_num
	);

	// updating biases, 'b'
	cublasSgemm_v2(
		cublas, CUBLAS_OP_N, CUBLAS_OP_N,
		neuron_num, 1, batch_size,
		&learning_rate,
		gradient, neuron_num,
		onevec, batch_size,
		&alpha,
		b, neuron_num
	);
}