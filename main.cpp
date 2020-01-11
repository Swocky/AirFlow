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
#include "utils/helper.cuh"
#include "utils/helper_host.hpp"
#include "layer/vgg16.hpp"
#include "layer/LeNet.hpp"

cudnnHandle_t cudnn; // global cudnn Handle
cublasHandle_t cublas; // global cublas Handle

float learning_rate = 0.01f; // learning_rate throughout the whole network
							  // can be change by UpdateLR func.
int main(){
	checkCUDNN(cudnnCreate(&cudnn)); // Initializing cudnn Handle
	cublasCreate_v2(&cublas);  // Initializing cublas Handle

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
	unsigned const batch_size = 64;
	learning_rate /= batch_size;

	unsigned const class_num = 10;
	unsigned const imageX = 32, imageY = 32;
	unsigned const channel_num = 3;

	//VGG16 vgg16(
	//	train_data,
	//	train_labels,
	//	test_data,
	//	test_labels,
	//	class_num,
	//	channel_num,
	//	imageX, imageY,
	//	batch_size
	//);

	//// Training Network for some number of epochs
	//vgg16.train(10000);

	LeNet lenet(
		train_data,
		train_labels,
		test_data,
		test_labels,
		class_num,
		channel_num,
		imageX, imageY,
		batch_size
	);

	// Training Network for some number of epochs
	lenet.train(10000);

	std::cout << "Press Enter to continue...\n";
	std::cin.get();
}