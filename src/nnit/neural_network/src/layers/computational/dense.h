#ifndef __DENSE_H__
#define __DENSE_H__

#include "computational.h"
#include "perceptron.h"

#include <vector>
#include <array>

namespace sixtron
{
using namespace std;

template <int INPUT, int OUTPUT>
class Dense: public Computational<INPUT, OUTPUT>
{
public:
    Dense(void) : Computational<INPUT, OUTPUT>() {
        for (uint32_t i = 0; i < OUTPUT; i++) {
            Perceptron<INPUT> perceptron;
            _perceptrons.push_back(perceptron);
        }
    }
    ~Dense(void) {}

    void load_weight(array<int8_t, INPUT * OUTPUT> weight) {
        array<int8_t, INPUT> perceptron_weight;
        for (unsigned int i = 0; i < INPUT * OUTPUT; i++) {
            perceptron_weight[i % INPUT] = weight[i];
            if (i % INPUT == INPUT - 1) {
                _perceptrons[int(i / INPUT)].load_weight(perceptron_weight);
            }
        }
    }

    array<int8_t, INPUT * OUTPUT> get_weight(void) {
        for (unsigned int i = 0; i < _perceptrons.size(); i++) {
            array<int8_t, INPUT> weight = _perceptrons[i].get_weight();
            for (unsigned int j = 0; j < INPUT; j++) {
                this->_weight[i * INPUT + j] = weight[j];
            }
        }

        return this->_weight;
    }

    void load_bias(array<int8_t, OUTPUT> bias) {
        for (unsigned int i = 0; i < _perceptrons.size(); i++) {
            _perceptrons[i].load_bias(array<int8_t, 1> {bias[i]}); // /!\ Cast to be reworked
        }
    }

    array<int8_t, OUTPUT> get_bias(void) {
        for (unsigned int i = 0; i < _perceptrons.size(); i++) {
            this->_bias[i] = _perceptrons[i].get_bias()[0];
        }

        return this->_bias;
    }
    
    array<int8_t, OUTPUT> forward(array<int8_t, INPUT> input) {
        for (unsigned int i = 0; i < _perceptrons.size(); i++) {
            this->_output[i] = _perceptrons[i].forward(input)[0];
        }

        return this->_output;
    }

private:
    std::vector<Perceptron<INPUT>> _perceptrons;
};

} // namespace sixtron

#endif // __DENSE_H__