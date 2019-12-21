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

class CNN {
public:
	CNN(std::vector<std::vector<float>>& _train_data,
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

	~CNN();

	void train(unsigned const epoch);
	void test();
};
