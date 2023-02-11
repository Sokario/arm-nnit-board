#ifndef __NEURAL_NETWORK_H__
#define __NEURAL_NETWORK_H__

#include "mbed.h"
#include "layer.h"

#include <vector>

namespace sixtron
{

class NNIT
{
public:
    NNIT();
    ~NNIT();

    void add_layer(Layer layer);
    void add_layers(std::vector<Layer> layers);
    void remove_layer(uint8_t index);

private:
    std::vector<uint8_t> input;
    std::vector<uint8_t> output;

    std::vector<Layer> layers;
};

} // namespace sixtron

#endif // __NEURAL_NETWORK_H__