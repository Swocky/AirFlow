#include <vector>

#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cudnn.h>
#include <ctime>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils/cifar10_reader.hpp"
#include "layer/convolutional_layer.hpp"
#include "layer/cnn.hpp"
#include "utils/helper.cuh"
#include "utils/helper_host.hpp"

cudnnHandle_t cudnn; // global cudnn Handle
cublasHandle_t cublas; // global cublas Handle

float learning_rate = 0.001f; // learning_rate throughout the whole network
							  // can be change by UpdateLR func.
int main(){
	checkCUDNN(cudnnCreate(&cudnn)); // Initializing cudnn Handle
	cublasCreate_v2(&cublas);  // Initializing cublas Handle

	// Usage...
	// 1. fill in 2D vector<float> train_data and test_data with values...
	//		memory layout should be CHW,
	//		for example;
	//		H*W red values, H*W green values, H*W blue values.
	//
	// 2. fill in 1d vector<unsigned> labels
	//		which go from 0 to class_num - 1.
	//		num of label elements must be equal to the corresponding data...
	//
	// 3. give values to necessary variables like batch_size or channel_num of the input image
	//
	// here as an example cifar-10 dataset is used.
	// download dataset https://www.cs.toronto.edu/~kriz/cifar.html, and put the files into a folder named 'cifar' relative to project's folder.
	// filling the values of train_data, test_data, train_labels, test_labels are filled by functions in ReadCifar10.hpp file...


	std::vector<std::vector<float>> train_data; train_data.reserve(40000);
	std::vector<unsigned> train_labels; train_labels.reserve(40000);
	std::vector<std::vector<float>> test_data; test_data.reserve(10000);
	std::vector<unsigned> test_labels; test_labels.reserve(10000);

	createTrainData(train_data, train_labels);
	createTestData(test_data, test_labels);
	std::cout << "Train data num: " << train_data.size() << '\n';
	std::cout << "Test data num: " << test_data.size() << '\n';

	/// creating architecture of the CNN
	// Necessary variables...
	std::vector<unsigned> kernel_nums = { 64, 64, 32 };
	std::vector<unsigned> kernel_sizes = { 4, 4, 3 };
	std::vector<unsigned> strides = { 1, 1, 1 };
	std::vector<unsigned> neuron_nums = { 1024 };
	unsigned const batch_size = 32;

	learning_rate /= batch_size;

	unsigned const class_num = 10;
	unsigned const imageX = 32, imageY = 32;
	unsigned const channel_num = 3;

	// initing cnn object & allocating needed memories on the gpu and randomizing weights and biases...
	CNN cnn(
		train_data,
		train_labels,
		test_data,
		test_labels,
		class_num,
		channel_num,
		imageX, imageY,
		kernel_nums,
		kernel_sizes,
		strides,
		neuron_nums,
		batch_size
	);

	// Training Network for some number of epochs
	cnn.train(10000);

	std::cout << "Press Enter to continue...\n";
	std::cin.get();
}