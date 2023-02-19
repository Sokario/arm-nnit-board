#ifndef __CONV2D_H__
#define __CONV2D_H__

#include "computational.h"
#include "convolution.h"

#include <array>

namespace sixtron
{

#define OUTPUT_WIDTH (((INPUT_WIDTH - KERNEL_WIDTH) / STRIDE_WIDTH) + 1)
#define OUTPUT_HEIGHT (((INPUT_HEIGHT - KERNEL_HEIGHT) / STRIDE_HEIGHT) + 1)

template <int INPUT_WIDTH, int INPUT_HEIGHT, int CHANNEL, int KERNEL_WIDTH, int KERNEL_HEIGHT, int FILTER, int STRIDE_WIDTH, int STRIDE_HEIGHT>
class Conv2D : public Computational<INPUT_WIDTH * INPUT_HEIGHT * CHANNEL, KERNEL_WIDTH * KERNEL_HEIGHT * FILTER, FILTER, OUTPUT_WIDTH * OUTPUT_HEIGHT * FILTER>
{
public:
    Conv2D() : Computational<INPUT_WIDTH * INPUT_HEIGHT * CHANNEL, KERNEL_WIDTH * KERNEL_HEIGHT * FILTER, FILTER, OUTPUT_WIDTH * OUTPUT_HEIGHT * FILTER>() {
        for (int i = 0; i < FILTER; i++) {
            Convolution<INPUT_WIDTH, INPUT_HEIGHT, KERNEL_WIDTH, KERNEL_HEIGHT, STRIDE_WIDTH, STRIDE_HEIGHT> convolution;
            _convolutions[i] = convolution;
        }
    }
    ~Conv2D() {}

    void load_weight(array<int8_t, KERNEL_WIDTH * KERNEL_HEIGHT * FILTER> weight) {
        array<int8_t, KERNEL_WIDTH * KERNEL_HEIGHT> convolution_weight;
        
        for (int i = 0; i < KERNEL_WIDTH * KERNEL_HEIGHT * FILTER; i++) {
            convolution_weight[i % (KERNEL_WIDTH * KERNEL_HEIGHT)] = weight[i];
            if (i % (KERNEL_WIDTH * KERNEL_HEIGHT) == KERNEL_WIDTH * KERNEL_HEIGHT - 1) {
                _convolutions[int(i / (KERNEL_WIDTH * KERNEL_HEIGHT))].load_weight(convolution_weight);
            }
        }
    }

    array<int8_t, KERNEL_WIDTH * KERNEL_HEIGHT * FILTER> get_weight(void) {
        for (int i = 0; i < FILTER; i++) {
            array<int8_t, KERNEL_WIDTH * KERNEL_HEIGHT> weight = _convolutions[i].get_weight();
            for (unsigned int j = 0; j < KERNEL_WIDTH * KERNEL_HEIGHT; j++) {
                this->_weight[i * KERNEL_WIDTH * KERNEL_HEIGHT + j] = weight[j];
            }
        }

        return this->_weight;
    }

    void load_bias(array<int8_t, FILTER> bias) {
        for (int i = 0; i < FILTER; i++) {
            _convolutions[i].load_bias(array<int8_t, 1> {bias[i]});
        }
    }

    array<int8_t, FILTER> get_bias(void) {
        for (int i = 0; i < FILTER; i++) {
            this->_bias[i] = _convolutions[i].get_bias()[0];
        }

        return this->_bias;
    }
    
    int8_t* forward(const int8_t* input) {
        for (int i = 0; i < FILTER; i++) {
            array<int8_t, INPUT_WIDTH * INPUT_HEIGHT> channel;
            int8_t value[OUTPUT_WIDTH * OUTPUT_HEIGHT] = {};
            int8_t* output = value;

            for (int j = 0; j < INPUT_WIDTH * INPUT_HEIGHT * CHANNEL; j++) {
                channel[j % (INPUT_WIDTH * INPUT_HEIGHT)] = input[j];
                if (j % (INPUT_WIDTH * INPUT_HEIGHT) == INPUT_WIDTH * INPUT_HEIGHT - 1) {
                    output = _convolutions[i].add_forward(channel.data(), output);
                }
            }

            // /!\ Better use memcpy
            for (int j = 0; j < OUTPUT_WIDTH * OUTPUT_HEIGHT; j++) {
                this->_output[i * OUTPUT_WIDTH * OUTPUT_HEIGHT + j] = output[j];
            }
        }

        return this->_output.data();
    }

private:
    array<Convolution<INPUT_WIDTH, INPUT_HEIGHT, KERNEL_WIDTH, KERNEL_HEIGHT, STRIDE_WIDTH, STRIDE_HEIGHT>, FILTER> _convolutions;
};

} // namespace sixtron


#endif // __CONV2D_H__