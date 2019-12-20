#include "helper.cuh"

// the utils function of print array
void PrintArrayAPI(float* Arr, unsigned n)
{
    for (unsigned i = 0; i != n; i++){
		printf("%f ", float(Arr[i]));
	}
	printf("\n");
}

// the kernel function of computing back propagation in Softmax layer
__global__ void SoftmaxLossBackprop(
	const float* result, float* error,
	unsigned const classCount, unsigned const batch_size,
	const unsigned* label_ptr)
{
	unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    // using one single line to avoid the divergence
    error[idx] = 1.0f * (threadIdx.x == label_ptr[blockIdx.x]) - result[idx];
}

// the utils function of computing back propagation in Softmax layer
void SoftmaxLossBackpropAPI(
	const float* result, float* error,
	unsigned const classCount, unsigned const batch_size,
	const unsigned* label_ptr)
{
	SoftmaxLossBackprop << < batch_size, classCount >> > (result, error, classCount, batch_size, label_ptr);
	cudaDeviceSynchronize();
}





