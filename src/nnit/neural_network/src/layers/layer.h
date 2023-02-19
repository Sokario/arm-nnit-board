#ifndef __LAYER_H__
#define __LAYER_H__

#include "mbed.h"

#include <array>

namespace sixtron
{
using namespace std;

class Layer
{
public:
    Layer(int input_size, int output_size) : _input_size(input_size), _output_size(output_size) {}
    ~Layer(void) {}

    // Computational
    //virtual void load_weight(array<int8_t, OUTPUT> weight) { return; }
    //virtual array<int8_t, OUTPUT> get_weight(void) { return {}; }
    //virtual void load_bias(array<int8_t, OUTPUT> bias) { return; }
    //virtual array<int8_t, OUTPUT> get_bias(void) { return {}; }

    // Activation
    virtual int8_t* derivative(const int8_t* error) { return {}; }

    // Commun
    int input_size(void) { return _input_size; }
    int output_size(void) { return _output_size; }
    virtual int8_t* forward(const int8_t* input) = 0;

protected:
    int _input_size;
    int _output_size;
};

} // namespace sixtron


#endif // __LAYER_H__