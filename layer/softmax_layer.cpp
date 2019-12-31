#include "softmax_layer.hpp"

SoftmaxLayer::SoftmaxLayer() {} // default

void SoftmaxLayer::init(unsigned const _class_num,
	                    unsigned const _input_num,
	                    float* _x,
	                    float* _p_gradient,
	                    unsigned const _batch_size){
	class_num = _class_num;
	input_num = _input_num;
	x = _x;
	p_gradient = _p_gradient;
	batch_size = _batch_size;

	// allocating weights, 'w'
	size_t const w_bytes = input_num * class_num * sizeof(float);
	checkCuda(cudaMalloc(&w, w_bytes));
	randomize(w, w_bytes / sizeof(float), 0.04f);

	// allocating biases, 'b' & creating and setting b_desc
	size_t const b_bytes = class_num * sizeof(float);
	checkCuda(cudaMalloc(&b, b_bytes));
	randomize(b, class_num, 0.01f);
	checkCUDNN(cudnnCreateTensorDescriptor(&b_desc));
	checkCUDNN(cudnnSetTensor4dDescriptor(
		b_desc,
		CUDNN_TENSOR_NCHW,
		CUDNN_DATA_FLOAT,
		1,
		class_num,
		1, 1
	));

	// allocating output before softmax, 'o' & softmax, 'y' & creating and setting y_desc
	size_t const y_bytes = class_num * batch_size * sizeof(float);
	checkCuda(cudaMalloc(&o, y_bytes));
	checkCuda(cudaMalloc(&y, y_bytes));
	checkCUDNN(cudnnCreateTensorDescriptor(&y_desc));
	checkCUDNN(cudnnSetTensor4dDescriptor(
		y_desc,
		CUDNN_TENSOR_NCHW,
		CUDNN_DATA_FLOAT,
		batch_size,
		class_num,
		1, 1
	));

	// allocating gradient
	checkCuda(cudaMalloc(&gradient, y_bytes));

	// allocating onevec
	checkCuda(cudaMalloc(&onevec, batch_size * sizeof(float)));
	std::vector<float> temp_onevec(batch_size, 1.0f);
	checkCuda(cudaMemcpy(onevec, temp_onevec.data(), batch_size * sizeof(float), cudaMemcpyHostToDevice));
}

SoftmaxLayer::~SoftmaxLayer(){
	
	/*cudaFree(y);
	cudaFree(w);
	cudaFree(b);
	cudaFree(o);
	cudaFree(gradient);
	checkCUDNN(cudnnDestroyTensorDescriptor(y_desc));
	checkCUDNN(cudnnDestroyTensorDescriptor(b_desc));*/
}

void SoftmaxLayer::feedForward()
{
	const float alpha = 1.0f, beta = 0.0f;

	// multiplying by weights, o = x*w;
	cublasSgemm_v2(
		cublas, CUBLAS_OP_N, CUBLAS_OP_N,
		class_num, batch_size, input_num,
		&alpha,
		w, class_num,
		x, input_num,
		&beta,
		o, class_num
	);

	// adding biases, o += b;
	checkCUDNN(cudnnAddTensor(
		cudnn,
		&alpha,
		b_desc, b,
		&alpha,
		y_desc, o
	));

	checkCUDNN(cudnnSoftmaxForward(
		cudnn,
		CUDNN_SOFTMAX_FAST,
		CUDNN_SOFTMAX_MODE_CHANNEL,
		&alpha,
		y_desc, o,
		&beta,
		y_desc, y
	));
}

void SoftmaxLayer::backprop()
{
	const float alpha = 1.0f, beta = 0.0f;
	//const float learning_rate_l = learning_rate / batch_size;
	const float learning_rate_l = learning_rate;

	// passing gradient to previous layer
	cublasSgemm_v2(
		cublas, CUBLAS_OP_T, CUBLAS_OP_N,
		input_num, batch_size, class_num,
		&alpha,
		w, class_num,
		gradient, class_num,
		&beta,
		p_gradient, input_num
	);

	// updating weights, 'w'
	cublasSgemm_v2(
		cublas, CUBLAS_OP_N, CUBLAS_OP_T,
		class_num, input_num, batch_size,
		&learning_rate_l,
		gradient, class_num,
		x, input_num,
		&alpha,
		w, class_num
	);

	// updating biases, 'b'
	cublasSgemm_v2(
		cublas, CUBLAS_OP_N, CUBLAS_OP_N,
		class_num, 1, batch_size,
		&learning_rate_l,
		gradient, class_num,
		onevec, batch_size,
		&alpha,
		b, class_num
	);
}
