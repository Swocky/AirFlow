#include "pooling_layer.hpp"

PoolingLayer::PoolingLayer() {}

void PoolingLayer::init(unsigned const _pooling_size,
                        unsigned const _input_num,
		                float* _x,
                        float* _p_gradient,
		                unsigned const _batch_size) {
    pooling_size = _pooling_size;
    input_num = _input_num;
    x = _x;
    p_gradient = _p_gradient;
    batch_size = _batch_size;


}
    
void PoolingLayer::~PoolingLayer(){}

void PoolingLayer::feedForward() {
    
}

void PoolingLayer::backprop() {
    const float alpha = 1.0f, beta = 0.0f;
    checkCUDNN(cudnnPoolingBackward(
        cudnn,
        activation_desc,
        &alpha,
        y_desc, y,
        y_desc, gradient,
        y_desc, o,
        &beta,
        y_desc, gradient
    ));
}