#include "cnn.hpp"

CNN::CNN(
	std::vector<std::vector<float>>& _train_data,
	std::vector<unsigned>& _train_labels,
	std::vector<std::vector<float>>& _test_data,
	std::vector<unsigned>& _test_labels,
	const unsigned _class_num,
	const unsigned _channel_num,
	const unsigned _image_x,
	const unsigned _image_y,
	std::vector<unsigned>& _kernel_nums,
	std::vector<unsigned>& _kernel_nums,
	std::vector<unsigned>& _strides,
	std::vector<unsigned>& _neuron_nums,
	const unsigned _batch_size
):
	train_data(_train_data),
	train_labels(_train_labels),
	test_data(_test_data),
	test_labels(_test_labels),
	class_num(_class_num),
	channel_num(_channel_num),
	image_x(_image_x),
	image_y(_image_y),
	kernel_nums(_kernel_nums),
	kernel_sizes(_kernel_sizes),
	strides(_strides),
	neuron_nums(_neuron_nums),
	train_data_num(_train_data.size()),
	test_data_num(_test_data.size()),
	input_size(_channel_num * _image_x * _image_y),
	convolutional_layer_num(_kernel_nums.size()),
	batch_size(_batch_size)
{
    // output parameters of the CNN
    std::cout<<"CNN parameters------------------"<<endl;
    std::cout<<"input size: "<<input_size<<endl;
	std::cout<<"batch size: "<<batch_size<<endl;
	std::cout<<"train data number: "<<train_data_mum<<endl;
	std::cout<<"kernel_size: "<<kernel_sizes<<endl;
	std::cout<<"kernel numbers: "<<kernel_nums<<endl;
	std::cout<<"image x: "<<image_x<<endl;
	std::cout<<"image y: "<<image_y<<endl;
	std::cout<<"class_numbers: "<<class_num<<endl;
    std::cout<<"--------------------------------"<<endl;

	// allocate and copy to dev_train_data
	size_t const train_data_bytes = train_data_num
							 * input_size * sizeof(float);
	checkCuda(cudaMalloc(&dev_train_data, train_data_bytes));
	for (unsigned i = 0; i != train_data_num; i++) {
		checkCuda(cudaMemcpy(dev_train_data + i * input_size,
			 train_data[i].data(), input_size * sizeof(float),
			 						 cudaMemcpyHostToDevice));
	}

	// allocate and copy to dev_train_labels
	size_t const train_labels_bytes = train_data_num
										 * sizeof(unsigned);
	checkCuda(cudaMalloc(&dev_train_labels, train_labels_bytes));
	checkCuda(cudaMemcpy(dev_train_labels, train_labels.data(),
					 train_labels_bytes, cudaMemcpyHostToDevice));

	// allocate and copy to dev_test_data
	size_t const test_data_bytes = test_data_num
								 * input_size * sizeof(float);
	checkCuda(cudaMalloc(&dev_test_data, test_data_bytes));
	for (unsigned i = 0; i != test_data_num; i++) {
		checkCuda(cudaMemcpy(dev_test_data + i * input_size,
					 					test_data[i].data(),
					  			  input_size * sizeof(float),
					 				cudaMemcpyHostToDevice));
	}

	// allocate and copy to dev_test_labels
	size_t const test_labels_bytes = test_data_num
										 * sizeof(unsigned);
	checkCuda(cudaMalloc(&dev_test_labels, test_labels_bytes));
	checkCuda(cudaMemcpy(dev_test_labels, test_labels.data(),
							 		test_labels_bytes,
							 		cudaMemcpyHostToDevice));
	
	float* dummy_gradient{ nullptr };
	size_t const dummy_gradient_bytes = batch_size
								 * input_size * sizeof(float);
	checkCuda(cudaMalloc(&dummy_gradient, dummy_gradient_bytes));
	// setting up the convolutional layers
	for (unsigned i = 0; i != convolutional_layer_num; i++) {
		if (i == 0) {
			convLayers.push_back(ConvolutionalLayer(
				kernel_nums[i],
				kernel_sizes[i],
				strides[i],
				channel_num,
				image_x, image_y,
				dev_train_data,
				dummy_gradient,
				batch_size
			));
			// dummy gradient is used because in this version,
			// there is no way to specify the first layer of 
			// the convolutional layer...
			// when using ConvolutionalLayer::BackProp(), all 
			// layers propagate data back to previous layer.
		} else {
			convLayers.push_back(ConvolutionalLayer(
				kernel_nums[i],
				kernel_sizes[i],
				strides[i],
				kernel_nums[i - 1],
				convLayers.back().output_x,
				convLayers.back().output_y,
				convLayers.back().y,
				convLayers.back().gradient,
				batch_size
			));
		}
	}

	// init FullyConnected class object, which will use 
	// FullyConnectedLayer class to represent fully connected
	// neurons in LeNet
	fc.init(
		neuron_nums,
		convLayers.back().output_x
						* convLayers.back().output_y 
						* kernel_nums.back(),
		convLayers.back().y,
		convLayers.back().gradient,
		batch_size
	);

	// init the last layer of the network which is softmax
	softmaxLayer.init(
		class_num,
		neuron_nums.back(),
		fc.y,
		fc.gradient,
		batch_size
	);
}

CNN::~CNN(){}

// Train the network for some number of epochs...
void CNN::train(unsigned const epoch) {
	for (unsigned p = 0; p != epoch; p++) {
		std::cout<<"Epoch: "<<p<<endl;
		unsigned* label_ptr{ nullptr };
		unsigned error_num = 0;
		for (unsigned i = 0; i < (train_data_num / batch_size) 
						* batch_size - batch_size;
						 i += batch_size) {
			// set x for the network's input
			convLayers.front().set_x(dev_train_data + i * input_size);
			// adjust pointer for the inputs' labels
			label_ptr = dev_train_labels + i;

			// manually forwarding ConvolutionLayer class objects
			for (auto &c : convLayers) {
				c.convolutionForward();
			}
 			// manually forwarding FullyConnected class which 
			// forwards individual layers
			fc.forward();
			// manually forwarding softmaxLayer to achiev
			// network's answers...
			softmaxLayer.feedForward();

			// can be commented out if training error rate is
			// not wanted, lowers performance when enabled...
			for (unsigned j = 0; j != batch_size; j++) {
				int result = 0;
				cublasIsamax_v2(cublas, class_num, 
							softmaxLayer.y + j * class_num, 1, &result);
				// (result - 1) since cublasIsamax starts numing from 1.
				if (result - 1 != train_labels[i + j]) {
					error_num++;
				}
			}

			// for testing purposes if train_data is big and test
			// accuracy is wanted before a whole epoch...
			if (i % 4992 * 1 == 0 && i != 0) {
				test();
			}

			// backprop part
			// softmaxbackprop
			SoftmaxLossBackpropAPI(softmaxLayer.y, softmaxLayer.gradient,
										 class_num, batch_size, label_ptr);
			softmaxLayer.backprop();
			fc.backprop();
			for (int k = convLayers.size() - 1; k >= 0; k--) {
				convLayers[k].convolutionBackward();
			}
		}

		float error_percent = error_num / float(train_data_num) * 100.f;
		std::cout << "Train data Accuracy: " << 1 - error_percent 
			<< "\t Learning rate: " << learning_rate * batch_size <<endl;
		test();
	}
}

// Tests networks performance(accuracy) on the test_data
void CNN::test(){
	unsigned* label_ptr{ nullptr };
	unsigned error_num = 0;

	for (unsigned i = 0; i < (test_data_num / batch_size)
					 * batch_size - batch_size; i += batch_size) {
		//std::cout << "iteration is: " << i << std::endl;
		convLayers.front().set_x(dev_test_data + i * input_size);
		label_ptr = dev_test_labels + i;

		for (auto &c : convLayers){
			c.convolutionForward();
		}

		fc.forward();
		softmaxLayer.feedForward();
		for (unsigned j = 0; j != batch_size; j++) {
			int result = 0;
			cublasIsamax_v2(cublas, class_num,
						 softmaxLayer.y + j * class_num, 1, &result);
			if (result - 1 != test_labels[i + j]) {
				error_num++;
			}
		}
		// backpropagation part is not used, since we are only testing
	}
	float error_percent = error_num / float(test_data_num) * 100.f;
	std::cout << "Test data Error Rate: " << 1 - error_percent << 
				"\t Learning rate: " << learning_rate * batch_size <<endl;

	// learning_rate adjustments
	UpdateLR(&learning_rate, batch_size);
}