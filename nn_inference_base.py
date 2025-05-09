# Cong Tran 
# 1002046419

import numpy as np
import os
import sys
from nn_load import *
from nn_inference_solution import nn_inference


def print_nn_outputs(alphas, zs):
    print()
    for i in range(1, len(zs)):
        if (i == 1):
            print("Layer 1, no alpha values (input layer).")
        else:
            # print a values.
            print("layer %d, a values: [" % (i), end=" ");
            for j in range(0, alphas[i].shape[0]):
                print("%8.4f" % (alphas[i][j,0]), end=" ");
            print("]")
            
        # print z values.
        print("layer %d, z values: [" % (i), end=" ");
        for j in range(0, zs[i].shape[0]):
            print("%8.4f" % (zs[i][j,0]), end=" ");
        print("]\n")


# Specify parameters for a test case.
nn_file = "nn2.txt"
input_file = "input2b.txt"

#activation_string = "step"
activation_string = "sigmoid"

# Read the neural network topology and weight information from a file.
(layers, units, biases, weights) = nn_load(nn_file)
#print_nn_info(layers, units, biases, weights)    

# Read the input vector.
input_vector = read_matrix(input_file)

# This is where your function is called.
(alphas, zs) = nn_inference(layers, units, biases, weights, 
                            activation_string, input_vector)

# Print the results.
print()
print_nn_outputs(alphas, zs)


# MY CODE FOR TESTING

# test_case = [
#     ("nn_xor.txt", "input1_00.txt", "step"), 
#     ("nn_xor.txt", "input1_01.txt", "step"), 
#     ("nn_xor.txt", "input1_01.txt", "sigmoid"), 
#     ("nn2.txt", "input2a.txt", "step"), 
#     ("nn2.txt", "input2b.txt", "sigmoid"),
# ]

# for nn_file, i_file, act_function in test_case: 
#     (layers, units, biases, weights) = nn_load(nn_file)
#     input_vector = read_matrix(i_file)
#     (alphas, zs) = nn_inference(layers, units, biases, weights, 
#                             act_function, input_vector)
#     print()
#     print_nn_outputs(alphas, zs)