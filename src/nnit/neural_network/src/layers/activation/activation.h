#ifndef __ACTIVATION_H__
#define __ACTIVATION_H__

#include "layer.h"

#include <array>

namespace sixtron
{
using namespace std;

template <int INPUT>
class Activation: public Layer
{
public:
    Activation(void) : Layer(INPUT, INPUT) {
        _output.fill(0);
    }
    ~Activation(void) {}

    virtual int8_t* derivative(const int8_t* error) = 0;

protected:
    array<int8_t, INPUT> _output; // /!\ Initialize _output for warning clearance
};

} // namespace sixtron


#endif // __ACTIVATION_H__