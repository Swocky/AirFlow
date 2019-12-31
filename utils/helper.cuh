#pragma once
#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cudnn.h>
#include <ctime>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_host.hpp"

// the utils function of print array
template <typename T>
void printArrayAPI(T* arr, unsigned const n);

// the kernel function of computing back propagation in Softmax layer
__global__ void softmaxLossBackprop(
	const float* result, float* error,
	unsigned const class_num, unsigned const batch_size,
	const unsigned* label_ptr);

// the utils function of computing back propagation in Softmax layer
void softmaxLossBackpropAPI(
	const float* result, float* error,
	unsigned const class_num, unsigned const batch_size,
	const unsigned* label_ptr);
