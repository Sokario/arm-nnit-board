#ifndef __NEURAL_NETWORK_H__
#define __NEURAL_NETWORK_H__

#include "mbed.h"

#include "layer.h"
#include "conv2D.h"
#include "dense.h"
#include "relu.h"

#include <vector>
#include <cstring>

namespace sixtron
{
using namespace std;

template <int INPUT, int OUTPUT>
class NNIT : public Layer
{
public:
    NNIT(void) : Layer(INPUT, OUTPUT) {}
    ~NNIT(void) {}

    void add_layer(Layer* layer) {
        _layers.push_back(layer);
    }

    void add_layers(std::vector<Layer*> layers) {
        for (unsigned int i = 0; i < layers.size(); i++) {
            add_layer(layers[i]);
        }
    }

    void load_layer_weight(int index, const int8_t* weight) {
        _layers[index]->load_weight(weight);
    }

    void load_layer_bias(int index, const int8_t* bias) {
        _layers[index]->load_bias(bias);
    }

//    void remove_layer(uint8_t index);

    int8_t* forward(const int8_t* input) {
        int8_t* last_input = new int8_t[INPUT];
        memcpy(last_input, input, INPUT * sizeof(int8_t));

        for (unsigned int i = 0; i < _layers.size(); i++) {
            last_input = _layers[i]->forward(last_input);
        }

        return last_input;
    }

private:
    vector<Layer*> _layers;
};

} // namespace sixtron

#endif // __NEURAL_NETWORK_H__