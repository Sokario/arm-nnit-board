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
static const std::array<int8_t, 2*4> D_weight = {0, 1, 2, 3, 4, 5, 6, 7};
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
static NNIT<2*30, 4> multi_neural_network;

//void button_handler(void) {
//    event_flag.set(BUTTON_FLAG);
//}

void test_routine() {
    printf("INFERENCE!\n");
    //while (true) {
        //event_flag.wait_any(BUTTON_FLAG);
        printf("---------------------------------------------\n");
        printf("RELU INIT.. \n");
        int8_t* ReLU_output = relu.forward(ReLU_input.data());
        printf("ReLU size: [%d] | [%d]\n", relu.input_size(), relu.output_size());
        printf("ReLU Test: ");
        for (int i = 0; i < relu.output_size(); i++) {
            printf("[%d] ", ReLU_output[i]);
        }
        printf("\n\n");

        printf("NEURON INIT.. \n");
        perceptron.load_weight(N_weight.data());
        perceptron.load_bias(N_bias.data());
        int8_t* N_output = perceptron.forward(N_input.data());
        printf("NEURON size: [%d] | [%d]\n", perceptron.input_size(), perceptron.output_size());
        printf("NEURON Test: ");
        for (int i = 0; i < perceptron.output_size(); i++) {
            printf("[%d] ", N_output[i]);
        }
        printf("\n\n");

        printf("DENSE INIT.. \n");
        dense.load_weight(D_weight.data());
        dense.load_bias(D_bias.data());
        int8_t* D_output = dense.forward(D_input.data());
        printf("DENSE size: [%d] | [%d]\n", dense.input_size(), dense.output_size());
        printf("DENSE Test: ");
        for (int i = 0; i < dense.output_size(); i++) {
            printf("[%d] ", D_output[i]);
        }
        printf("\n\n");

        printf("CONV INIT.. \n");
        convolution.load_weight(C_weight.data());
        convolution.load_bias(C_bias.data());
        int8_t* C_output = convolution.forward(C_input.data());
        printf("CONV size: [%d] | [%d]\n", convolution.input_size(), convolution.output_size());
        printf("CONV Test: ");
        for (int i = 0; i < convolution.output_size(); i++) {
            printf("[%d] ", C_output[i]);
        }
        printf("\n\n");

        printf("CONV2D INIT.. \n");
        conv2D.load_weight(C2D_weight.data());
        conv2D.load_bias(C2D_bias.data());
        int8_t* C2D_output = conv2D.forward(C2D_input.data());
        printf("CONV2D size: [%d] | [%d]\n", conv2D.input_size(), conv2D.output_size());
        printf("CONV2D Test: ");
        for (int i = 0; i < conv2D.output_size(); i++) {
            printf("[%d] ", C2D_output[i]);
        }
        printf("\n");
        printf("---------------------------------------------\n");
        printf("NNIT INIT.. \n");
        conv2D.load_weight(C2D_weight.data());
        conv2D.load_bias(C2D_bias.data());
        neural_network.add_layer(&conv2D);
        int8_t* NN_output = neural_network.forward(C2D_input.data());
        printf("NNIT size: [%d] | [%d]\n", neural_network.input_size(), neural_network.output_size());
        printf("NNIT Test: ");
        for (int i = 0; i < neural_network.output_size(); i++) {
            printf("[%d] ", NN_output[i]);
        }
        printf("\n\n");

        printf("Multi NNIT INIT.. \n");
        multi_neural_network.add_layers({
            new Conv2D<5, 6, 1, 3, 3, 1, 1, 1>(),
            new Dense<12, 4>()
            });
        multi_neural_network.load_layer_weight(0, C_weight.data());
        multi_neural_network.load_layer_bias(0, C_bias.data());
        printf("LAYER LOADED!\n");
        static const std::array<int8_t, 4*12> MN_weight = {1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1};
        static const std::array<int8_t, 4> MN_bias = {0, 0, 0, 0};
        multi_neural_network.load_layer_weight(1, MN_weight.data());
        multi_neural_network.load_layer_bias(1, MN_bias.data());
        printf("LAYER LOADED!\n");
        int8_t* MNN_output = multi_neural_network.forward(C_input.data());
        printf("Multi NNIT size: [%d] | [%d]\n", multi_neural_network.input_size(), multi_neural_network.output_size());
        printf("Multi NNIT Test: ");
        for (int i = 0; i < multi_neural_network.output_size(); i++) {
            printf("[%d] ", MNN_output[i]);
        }
        printf("\n");
        printf("---------------------------------------------\n");
    //}
}

int main()
{
    printf("START!\n");
    //button.rise(&button_handler);
    thread.start(test_routine);

    while (true) {
        led1 = !led1;
        if (led1) {
            printf("Alive !\n");
        }
        ThisThread::sleep_for(HALF_PERIOD);
    }
}
