#ifndef __RELU_H__
#define __RELU_H__

#include "activation.h"

#include <cmath>
#include <array>

namespace sixtron
{
using namespace std;

template <int INPUT>
class ReLU: public Activation<INPUT>
{
public:
    ReLU(float scale_factor) : Activation<INPUT>() {
        set_factor(scale_factor);
    }

    ~ReLU(void) {}

    void set_factor(float scale_factor) {
        _factor = scale_factor;
    }
    
    array<int8_t, INPUT> derivative(array<int8_t, INPUT> error) {
        // /!\ To be sure on variable for check and return value
        for (int i = 0; i < INPUT; i++) {
            // /!\ 0 to be verified because of int8_t values
            this->_output[i] = error[i] * ((error[i] <= 0)? _factor : 1.0f);
        }

        return this->_output;
    }

    array<int8_t, INPUT> forward(array<int8_t, INPUT> input) {
        if (_factor == 0.0f) {
            for (int i = 0; i < INPUT; i++) {
                // /!\ 0 to be verified because of int8_t values
                this->_output[i] = std::max(float(input[i]), 0.0f);
            }
        } else {
            for (int i = 0; i < INPUT; i++) {
                // /!\ 0 to be verified because of int8_t values
                this->_output[i] = std::max(float(input[i]), _factor * float(input[i]));
            }
        }

        return this->_output;
    }

private:
    float _factor;
};

} // namespace sixtron

#endif // __RELU_H__