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
#include <fstream>

// get the cuda Handler from other
extern cudnnHandle_t cudnn;
extern cublasHandle_t cublas;

// get the learning rate to avoid duplicate definition
extern float learning_rate;

using namespace std;

// the 
#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      cerr << "CUDNN Error on file " << __FILE__ << " on line:  "      \
                << __LINE__ << ' ' \
				<< cudnnGetErrorString(status) << endl; \
    }                                                        \
  }



// macro for checking cuda errors
#define checkCuda(expression)										\
 {																	\
	cudaError_t err = (expression);									\
	if (err != cudaSuccess) {										\
		cerr << "Cuda Error on file " << __FILE__				\
				  << " on line: " << __LINE__ << ' '				\
				  << cudaGetErrorString(err) << endl;				\
	}																\
 }

// Randomizing 'n' number of floats, starting from 'data'
inline void Randomize(float* data, unsigned const n, const float stddev = 0.07f)
{
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937);
	curandSetPseudoRandomGeneratorSeed(gen, clock());
	curandGenerateNormal(gen, data, n, 0.0f, stddev);
}

// print one dimension vector
template <typename T>
std::ostream& operator<<(std::ostream& stream, const std::vector<T> vec)
{
	for (auto &i : vec){
		stream << i << ' ';
	}
	return (stream);
}

// to cout an array which resides in gpu ram
extern void ReadArrayAPI(float* Arr, unsigned const n);

// adjust learning_rate
inline void UpdateLR(float* learning_rate, unsigned const batch_size)
{
	*learning_rate = 0.001
	*learning_rate /= batch_size;
}