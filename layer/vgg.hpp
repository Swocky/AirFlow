#pragma once
#include <ctime>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cudnn.h>
#include <curand.h>
#include <iostream>
#include <vector>

#include "convolutional_layer.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "fc.hpp"
#include "../utils/helper.cuh"
#include "../utils/helper_host.hpp"
#include "softmax_layer.hpp"
#include "pooling_layer.hpp"

class VGG{
public:
    VGG(std::vector<std::vector<float>>& _train_data,
		std::vector<unsigned>& _train_labels,
		std::vector<std::vector<float>>& _test_data,
		std::vector<unsigned>& _test_labels,
		const unsigned _class_num,
		const unsigned _channel_num,
		const unsigned _image_x,
		const unsigned _image_y,
		std::vector<unsigned>& _kernel_nums,
		std::vector<unsigned>& _kernel_sizes,
		std::vector<unsigned>& _strides,
		std::vector<unsigned>& _neuron_nums,
		const unsigned _batch_size
    );

    ~VGG();

    void train(unsigned const epoch);
    float test();

private:
    // definining vars of the vgg
	unsigned const class_num;
	unsigned const channel_num;
	const unsigned image_x;
	const unsigned image_y;
	std::vector<unsigned>& kernel_nums;
	std::vector<unsigned>& kernel_sizes;
	std::vector<unsigned>& strides;
	std::vector<unsigned>& neuron_nums;
    const unsigned batch_size;

    // understood varibles
	const unsigned train_data_num;
	const unsigned test_data_num;
	const unsigned input_size;
	const unsigned convolutional_layer_num;

    // host data
	std::vector<std::vector<float>>& train_data;
	std::vector<unsigned>& train_labels;
	std::vector<std::vector<float>>& test_data;
	std::vector<unsigned>& test_labels;

    // device data ptrs
	float* dev_train_data{ nullptr };
	unsigned* dev_train_labels{ nullptr };
	float* dev_test_data{ nullptr };
	unsigned* dev_test_labels{ nullptr };

    // define layers
    ConvolutionalLayer conv1 = NULL;
    PoolingLayer pool1 = NULL;
    ConvolutionalLayer conv2 = NULL;
    PoolingLayer pool2 = NULL;
    FullyConnected fc;
    SoftmaxLayer softmax_layer;
}