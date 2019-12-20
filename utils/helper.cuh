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
#include "helper_host.h"

// the utils function of print array
template <typename T>
void PrintArrayAPI(T* Arr, unsigned const n);

// the kernel function of computing back propagation in Softmax layer
__global__ void SoftmaxLossBackprop(
	const float* result, float* error,
	unsigned const classCount, unsigned const batch_size,
	const unsigned* label_ptr);

// the utils function of computing back propagation in Softmax layer
void SoftmaxLossBackpropAPI(
	const float* result, float* error,
	unsigned const classCount, unsigned const batch_size,
	const unsigned* label_ptr);
