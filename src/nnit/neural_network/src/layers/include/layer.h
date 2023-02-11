#ifndef __LAYER_H__
#define __LAYER_H__

#include "mbed.h"

#include <vector>

namespace sixtron
{

class Layer
{
public:
    Layer(/* args */);
    ~Layer();

private:
    // /!\ TO DO: to be added to the flash memory
    std::vector<uint8_t> data;
};

} // namespace sixtron


#endif // __LAYER_H__