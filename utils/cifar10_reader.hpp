#pragma once
#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <fstream>

// the function of reading files
void readFiles(std::string file_name,
			   std::vector<std::vector<float>>& data,
			   std::vector<unsigned>& data_labels){
	std::ifstream binary_file(file_name, std::ios::binary | std::ios::in);
	unsigned image_size = 3073;
	std::unique_ptr<unsigned char[]> image_ptr(new unsigned char[image_size]);

	if (binary_file.is_open()){
		while (binary_file.read(reinterpret_cast<char*>(image_ptr.get()),
								image_size)){
			// assigning labels
			data_labels.push_back(image_ptr.get()[0]);

			// assigning data
			std::vector<float> temp_data(3072);
			for (unsigned elem = 0; elem != temp_data.size(); elem++){
				temp_data[elem] = float(image_ptr.get()[elem + 1])
									 / 255.0f - 0.5f;
			}
			data.push_back(temp_data);
		}
	}
	else {
		std::cout << "Error opening the file... " << std::endl;
		std::cin.get();
	}
}

// the function of reading train file
void createTrainData(std::vector<std::vector<float>>& train_data,
					 std::vector<unsigned>& train_data_labels){
	readFiles("cifar/data_batch_1.bin", train_data, train_data_labels);
	readFiles("cifar/data_batch_2.bin", train_data, train_data_labels);
	readFiles("cifar/data_batch_3.bin", train_data, train_data_labels);
	readFiles("cifar/data_batch_4.bin", train_data, train_data_labels);
}

void createTestData(std::vector<std::vector<float>>& test_data,
					std::vector<unsigned>& test_data_labels){
	readFiles("cifar/test_batch.bin", test_data, test_data_labels);
}