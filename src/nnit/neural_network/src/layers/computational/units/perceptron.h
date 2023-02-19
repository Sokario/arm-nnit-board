#ifndef __PERCEPTRON_H__
#define __PERCEPTRON_H__

#include "computational.h"

#include <array>

namespace sixtron
{
using namespace std;

#define STATIC_ASSERT( condition )\
    typedef char assert_failed_condition [ (condition) ? 1 : -1 ];

template <int INPUT>
class Perceptron: public Computational<INPUT, INPUT, 1, 1>
{
public:
    Perceptron(void) : Computational<INPUT, INPUT, 1, 1>() {}
    ~Perceptron(void) {}

    array<int8_t, 1> forward(array<int8_t, INPUT> input) {
        //STATIC_ASSERT(sizeof(*input) == sizeof(*_weight));
        assert(sizeof(this->_bias) == sizeof(int8_t));
        
        this->_output[0] = 0;
        for (int i = 0; i < INPUT; i++) {
            this->_output[0] += input[i] * this->_weight[i];
        }
        this->_output[0] += this->_bias[0];

        return this->_output;
    }

private:
    // /!\ TO DO: to be added to the flash memory
    //int8_t _weight[INPUT];
    //int8_t _bias;
};

} // namespace sixtron


#endif // __PERCEPTRON_H__