#ifndef __LAYER_H__
#define __LAYER_H__

#include "mbed.h"

#include <array>

namespace sixtron
{
using namespace std;

template <int OUTPUT>
class Layer
{
public:
    Layer(void) {}
    ~Layer(void) {}

    // Computational
    //virtual void load_weight(array<int8_t, OUTPUT> weight) { return; }
    //virtual array<int8_t, OUTPUT> get_weight(void) { return {}; }
    //virtual void load_bias(array<int8_t, OUTPUT> bias) { return; }
    //virtual array<int8_t, OUTPUT> get_bias(void) { return {}; }

    // Activation
    virtual array<int8_t, OUTPUT> derivative(array<int8_t, OUTPUT> error) { return {}; }

    // Commun
    virtual array<int8_t, OUTPUT> forward(array<int8_t, OUTPUT> input) { return {}; }

protected:
    array<int8_t, OUTPUT> _output; // /!\ Initialize _output for warning clearance
};

} // namespace sixtron


#endif // __LAYER_H__