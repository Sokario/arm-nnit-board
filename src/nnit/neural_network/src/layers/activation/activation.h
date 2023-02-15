#ifndef __ACTIVATION_H__
#define __ACTIVATION_H__

#include "layer.h"

#include <array>

namespace sixtron
{
using namespace std;

template <int INPUT>
class Activation: public Layer<INPUT>
{
public:
    Activation(void) : Layer<INPUT>() {}
    ~Activation(void) {}

    virtual array<int8_t, INPUT> derivative(array<int8_t, INPUT> error) = 0;

};

} // namespace sixtron


#endif // __ACTIVATION_H__