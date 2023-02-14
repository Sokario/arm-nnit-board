#ifndef __COMPUTATIONAL_H__
#define __COMPUTATIONAL_H__

#include "layer.h"

#include <array>

namespace sixtron
{
using namespace std;

template <int INPUT, int OUTPUT>
class Computational: public Layer<OUTPUT>
{
public:
    Computational(void) : Layer<OUTPUT>() {}
    ~Computational(void) {}

    virtual void load_weight(array<int8_t, INPUT * OUTPUT> weight) {
        for (unsigned int i = 0; i < INPUT * OUTPUT; i++) {
            _weight[i] = weight[i];
        }
    }

    virtual array<int8_t, INPUT * OUTPUT> get_weight(void) { return _weight; }
    virtual void load_bias(array<int8_t, OUTPUT> bias) {
        for (unsigned int i = 0; i < OUTPUT; i++) {
            _bias[i] = bias[i];
        }
    }
    virtual array<int8_t, OUTPUT> get_bias(void) { return _bias; }

protected:
    // /!\ TO DO: to be added to the flash memory
    array<int8_t, INPUT * OUTPUT> _weight;
    array<int8_t, OUTPUT> _bias;
};

} // namespace sixtron


#endif // __COMPUTATIONAL_H__