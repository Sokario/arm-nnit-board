#ifndef __COMPUTATIONAL_H__
#define __COMPUTATIONAL_H__

#include "layer.h"

#include <array>

namespace sixtron
{
using namespace std;

template <int INPUT, int WEIGHT, int BIAS, int OUTPUT>
class Computational: public Layer
{
public:
    Computational(void) : Layer(INPUT, OUTPUT) {
        _weight.fill(0);
        _bias.fill(0);
        _output.fill(0);
    }
    ~Computational(void) {}

    virtual void load_weight(const int8_t* weight) {
        printf("LOADING WEIGHT: ");
        for (int i = 0; i < WEIGHT; i++) {
            _weight[i] = weight[i];
            printf("[%d]:[%d] ", weight[i], _weight[i]);
        }
        printf("\n");
    }
    virtual int8_t* get_weight(void) { return _weight.data(); }

    virtual void load_bias(const int8_t* bias) {
        for (int i = 0; i < BIAS; i++) {
            _bias[i] = bias[i];
        }
    }
    virtual int8_t* get_bias(void) { return _bias.data(); }

protected:
    // /!\ TO DO: to be added to the flash memory
    array<int8_t, WEIGHT> _weight;
    array<int8_t, BIAS> _bias;

    array<int8_t, OUTPUT> _output; // /!\ Initialize _output for warning clearance
};

} // namespace sixtron


#endif // __COMPUTATIONAL_H__