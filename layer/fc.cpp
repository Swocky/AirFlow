#include "fc.hpp"

void FullyConnected::init(const std::vector<unsigned> _neuron_counts,
	                      const unsigned _input_count,
	                      float* _x,
	                      float* _p_gradient,
	                      const unsigned _batch_size){
	// for initializer list
	neuron_counts = _neuron_counts;
	input_count =_input_count;
	x = _x;
	p_gradient = _p_gradient;
	batch_size = _batch_size;

	// creating fc_layers
	fc_layers = std::vector<FullyConnectedLayer>(neuron_counts.size());
	for (unsigned i = 0; i != neuron_counts.size(); i++){
		if (i == 0){
			fc_layers[i].init(
				neuron_counts[i],
				input_count,
				x,
				p_gradient,
				batch_size
			);
		}
		else {
			fc_layers[i].init(
				neuron_counts[i],
				neuron_counts[i - 1],
				fc_layers[i - 1].y,
				fc_layers[i - 1].gradient,
				batch_size
			);
		}
	}

	y = fc_layers.back().y;
	gradient = fc_layers.back().gradient;
}

// default constructor
FullyConnected::FullyConnected(){
	neuron_counts = std::vector<unsigned>();
}


FullyConnected::~FullyConnected(){
}

void FullyConnected::forward(){
	//ReadArrayAPI(fc_layers.back().w, neuron_counts.back());
	for (auto& i : fc_layers){
		i.feedForward();
	}
}

void FullyConnected::backprop(){
	for (int i = fc_layers.size() - 1; i >= 0; i--){
		fc_layers[i].backprop();
	}
}