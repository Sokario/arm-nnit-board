/*
 * Copyright (c) 2023, CATIE
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mbed.h"
#include "neural_network.h"

using namespace sixtron;

namespace {
#define HALF_PERIOD 500ms
}

#define BUTTON_FLAG 0x01 << 0
static InterruptIn button(BUTTON1);
static EventFlags event_flag;

Thread thread(osPriorityNormal);
static DigitalOut led1(LED1);

// ################# TESTs !!!
// ReLU
static const std::array<int8_t, 15> ReLU_input = {-127, -63, -31, -15, -7, -3, -1, 0, 1, 3, 7, 15, 31, 63, 127};
static ReLU<15> relu(0.0f); // {0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 7, 15, 31, 63, 127}
// NEURON
static const std::array<int8_t, 4> N_input = {1, 2, 7, 15};
static const std::array<int8_t, 4> N_weight = {0, 1, 3, 7};
static const std::array<int8_t, 1> N_bias = {1};
static Perceptron<4> perceptron; // {-127}
// DENSE
static const std::array<int8_t, 2> D_input = {1, 2};
static const std::array<int8_t, 8> D_weight = {0, 1, 2, 3, 4, 5, 6, 7};
static const std::array<int8_t, 4> D_bias = {1, 2, 3, 4};
static Dense<2, 4> dense; // {3, 10, 17, 24}
// CONVOLUTION
static const std::array<int8_t, 30> C_input = {25, 100, 75, 49, 0, 50, 80, 0, 70, 100, 5, 10, 20, 30, 0, 60, 50, 12, 24, 32, 37, 53, 55, 21, 90, 0, 17, 0, 23, 0};
static const std::array<int8_t, 9> C_weight = {1, 0, 0, 0, 1, 0, 1, 0, 1};
static const std::array<int8_t, 1> C_bias = {0};
static Convolution<5, 6, 3, 3, 1, 1> convolution; // {-126, -116, -91, -124, -82, 74, -109, 96, -67, 113, -111, 33}
// CONVOLUTION 2D
static const std::array<int8_t, 2*30> C2D_input = {25, 100, 75, 49, 0, 50, 80, 0, 70, 100, 5, 10, 20, 30, 0, 60, 50, 12, 24, 32, 37, 53, 55, 21, 90, 0, 17, 0, 23, 0, 25, 100, 75, 49, 0, 50, 80, 0, 70, 100, 5, 10, 20, 30, 0, 60, 50, 12, 24, 32, 37, 53, 55, 21, 90, 0, 17, 0, 23, 0};
static const std::array<int8_t, 2*9> C2D_weight = {1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1};
static const std::array<int8_t, 2*1> C2D_bias = {0, 0};
static Conv2D<5, 6, 2, 3, 3, 2, 1, 1> conv2D; // {-126, -116, -91, -124, -82, 74, -109, 96, -67, 113, -111, 33}
// NEURAL NETWORK
static NNIT<2*30, 2*12> neural_network;

void button_handler(void) {
    event_flag.set(BUTTON_FLAG);
}

void test_routine() {
    printf("INFERENCE!\n");
    while (true) {
        event_flag.wait_any(BUTTON_FLAG);
        printf("---------------------------------------------\n");
        printf("RELU INIT.. \n");
        std::array<int8_t, 15> ReLU_output = relu.forward(ReLU_input);
        printf("ReLU size: [%d] | [%d]\n", ReLU_input.size(), ReLU_output.size());
        printf("ReLU Test: ");
        for (unsigned int i = 0; i < ReLU_output.size(); i++) {
            printf("[%d] ", ReLU_output[i]);
        }
        printf("\n");

        printf("NEURON INIT.. \n");
        perceptron.load_weight(N_weight);
        perceptron.load_bias(N_bias);
        std::array<int8_t, 1> N_output = perceptron.forward(N_input);
        printf("NEURON size: [%d] | [%d]\n", N_input.size(), N_output.size());
        printf("NEURON Test: ");
        for (unsigned int i = 0; i < N_output.size(); i++) {
            printf("[%d] ", N_output[i]);
        }
        printf("\n\n");

        printf("DENSE INIT.. \n");
        dense.load_weight(D_weight);
        dense.load_bias(D_bias);
        std::array<int8_t, 4> D_output = dense.forward(D_input);
        printf("DENSE size: [%d] | [%d]\n", D_input.size(), D_output.size());
        printf("DENSE Test: ");
        for (unsigned int i = 0; i < D_output.size(); i++) {
            printf("[%d] ", D_output[i]);
        }
        printf("\n\n");

        printf("CONV INIT.. \n");
        convolution.load_weight(C_weight);
        convolution.load_bias(C_bias);
        std::array<int8_t, 12> C_output = convolution.forward(C_input);
        printf("CONV size: [%d] | [%d]\n", C_input.size(), C_output.size());
        printf("CONV Test: ");
        for (unsigned int i = 0; i < C_output.size(); i++) {
            printf("[%d] ", C_output[i]);
        }
        printf("\n\n");

        printf("CONV2D INIT.. \n");
        conv2D.load_weight(C2D_weight);
        conv2D.load_bias(C2D_bias);
        std::array<int8_t, 24> C2D_output = conv2D.forward(C2D_input);
        printf("CONV2D size: [%d] | [%d]\n", C2D_input.size(), C2D_output.size());
        printf("CONV2D Test: ");
        for (unsigned int i = 0; i < C2D_output.size(); i++) {
            printf("[%d] ", C2D_output[i]);
        }
        printf("\n");
        printf("---------------------------------------------\n");
        //printf("NNIT INIT.. \n");
        //conv2D.load_weight(C2D_weight);
        //conv2D.load_bias(C2D_bias);
        //neural_network.add_layer(&conv2D);
        //std::array<int8_t, 24> NN_output = neural_network.forward(C2D_input);
        //printf("NNIT size: [%d] | [%d]\n", C2D_input.size(), NN_output.size());
        //printf("NNIT Test: ");
        //for (unsigned int i = 0; i < NN_output.size(); i++) {
        //    printf("[%d] ", NN_output[i]);
        //}
        //printf("\n");
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
            printf("Alive !\n");
        }
        ThisThread::sleep_for(HALF_PERIOD);
    }
}
