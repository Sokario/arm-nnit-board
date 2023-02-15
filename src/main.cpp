/*
 * Copyright (c) 2023, CATIE
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mbed.h"
#include "neural_network.h"

using namespace sixtron;

namespace {
#define HALF_PERIOD 1s
}

#define BUTTON_FLAG 0x01 << 0
static InterruptIn button(BUTTON1);
static EventFlags event_flag;

Thread thread(osPriorityNormal);
static DigitalOut led1(LED1);

// ################# TESTs !!!
// ReLU
static std::array<int8_t, 15> ReLU_input = {-127, -63, -31, -15, -7, -3, -1, 0, 1, 3, 7, 15, 31, 63, 127};
static ReLU<15> relu(0.0f);
// NEURON
static std::array<int8_t, 4> N_input = {1, 2, 7, 15};
static std::array<int8_t, 4> N_weight = {0, 1, 3, 7};
static std::array<int8_t, 1> N_bias = {1};
static Perceptron<4> perceptron;
// DENSE
static std::array<int8_t, 2> D_input = {1, 2};
static std::array<int8_t, 8> D_weight = {0, 1, 2, 3, 4, 5, 6, 7};
static std::array<int8_t, 4> D_bias = {1, 2, 3, 4};
static Dense<2, 4> dense;

void button_handler(void) {
    event_flag.set(BUTTON_FLAG);
}

void test_routine() {
    printf("INFERENCE!\n");
    while (true) {
        event_flag.wait_any(BUTTON_FLAG);
        printf("---------------------------------------------\n");
        std::array<int8_t, 15> ReLU_output = relu.forward(ReLU_input);
        printf("ReLU size: [%d] | [%d]\n", ReLU_input.size(), ReLU_output.size());
        printf("ReLU Test: ");
        for (unsigned int i = 0; i < ReLU_output.size(); i++) {
            printf("[%d] ", ReLU_output[i]);
        }
        printf("\n");

        perceptron.load_weight(N_weight);
        perceptron.load_bias(N_bias);
        std::array<int8_t, 1> N_output = perceptron.forward(N_input);
        printf("NEURON size: [%d] | [%d]\n", N_input.size(), N_output.size());
        printf("NEURON Test: ");
        for (unsigned int i = 0; i < N_output.size(); i++) {
            printf("[%d] ", N_output[i]);
        }
        printf("\n");

        dense.load_weight(D_weight);
        dense.load_bias(D_bias);
        std::array<int8_t, 4> D_output = dense.forward(D_input);
        printf("DENSE size: [%d] | [%d]\n", D_input.size(), D_output.size());
        printf("DENSE Test: ");
        for (unsigned int i = 0; i < D_output.size(); i++) {
            printf("[%d] ", D_output[i]);
        }
        printf("\n");
        printf("---------------------------------------------\n");
    }
}

int main()
{
    printf("START!\n");
    button.rise(&button_handler);
    thread.start(test_routine);

    while (true) {
        led1 = !led1;
        if (led1) {
            printf("Alive!\n");
        }
        ThisThread::sleep_for(HALF_PERIOD);
    }
}
