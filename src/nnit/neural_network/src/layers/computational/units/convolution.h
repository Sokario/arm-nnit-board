#ifndef __CONVOLUTION_H__
#define __CONVOLUTION_H__

#include "computational.h"

namespace sixtron
{

#define OUTPUT_WIDTH (((INPUT_WIDTH - KERNEL_WIDTH) / STRIDE_WIDTH) + 1)
#define OUTPUT_HEIGHT (((INPUT_HEIGHT - KERNEL_HEIGHT) / STRIDE_HEIGHT) + 1)

template <int INPUT_WIDTH, int INPUT_HEIGHT, int KERNEL_WIDTH, int KERNEL_HEIGHT, int STRIDE_WIDTH, int STRIDE_HEIGHT>
class Convolution : public Computational<INPUT_WIDTH * INPUT_HEIGHT, KERNEL_WIDTH * KERNEL_HEIGHT, 1, OUTPUT_WIDTH * OUTPUT_HEIGHT>
{
public:
    Convolution(void) : Computational<INPUT_WIDTH * INPUT_HEIGHT, KERNEL_WIDTH * KERNEL_HEIGHT, 1, OUTPUT_WIDTH * OUTPUT_HEIGHT>() {}
    ~Convolution(void) {}

    int8_t* forward(const int8_t* input) {
        for (unsigned int i = 0; i < OUTPUT_WIDTH * OUTPUT_HEIGHT; i++) {
            this->_output[i] = 0;
            for (unsigned int j = 0; j < KERNEL_WIDTH * KERNEL_HEIGHT; j++) {
                this->_output[i] += this->_weight[j] * input[(i % OUTPUT_WIDTH) * STRIDE_WIDTH + (INPUT_WIDTH * int(i / OUTPUT_WIDTH)) * STRIDE_HEIGHT + (j % KERNEL_WIDTH) + (INPUT_WIDTH * int(j / KERNEL_WIDTH))];
            }
            this->_output[i] += this->_bias[0];
        }

        return this->_output.data();
    }

    int8_t* add_forward(const int8_t* input, const int8_t* previous) {
        for (unsigned int i = 0; i < OUTPUT_WIDTH * OUTPUT_HEIGHT; i++) {
            this->_output[i] = previous[i];
            for (unsigned int j = 0; j < KERNEL_WIDTH * KERNEL_HEIGHT; j++) {
                this->_output[i] += this->_weight[j] * input[(i % OUTPUT_WIDTH) * STRIDE_WIDTH + (INPUT_WIDTH * int(i / OUTPUT_WIDTH)) * STRIDE_HEIGHT + (j % KERNEL_WIDTH) + (INPUT_WIDTH * int(j / KERNEL_WIDTH))];
            }
            this->_output[i] += this->_bias[0];
        }

        return this->_output.data();
    }
};

} // namespace sixtron


#endif // __CONVOLUTION_H__