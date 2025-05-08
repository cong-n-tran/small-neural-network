# Cong Tran 
# 1002046419

import numpy as np

# layers: an integer, specifies the number of layers in the network.
# units: a list of integers specifying the number of units in each layer.
    # units[L] is the number of units in layer L (where layer 1 is the input layer).
    # units[0] = None, since in our notation there is no layer 0.
    # units[1] is the number of units in the input layer
    # units[layers] is the number of units in the output layer
# biases: a list of numpy column matrices. 
    # biases[L] contains the biases in layer L (where layer 1 is the input layer).
    # biases[0] = None, since in our notation there is no layer 0.
    # biases[1] = None, since the input layer contains no perceptrons and thus no bias values
    # biases[2] contains the biases in the first hidden layer
    # biases[layers] contains the biases in the output layer.
# weights: a list of numpy matrices specifying the weights in each layer.
    # weights[L] contains the weights in layer L (where layer 1 is the input layer).
        # Its number of rows is the number of units in layer L (saved as units[L])
        # Its number of columns is the number of units in layer L-1 (saved as units[L-1])
        # Every row specifies the weights in a unit of layer L
    # weights[0] = None, since in our notation there is no layer 0.
    # weights[1] = None, since the input layer contains no perceptrons and thus no weight values
    # weights[2] contains the weights in the first hidden layer
    # weights[layers-2] contains the biases in the output layer.
# activation: it is a string that specifies the activation function. The value is either "step" or "sigmoid".
# input_vector: a column vector specifying the input to the perceptron. It is a 2D numpy array with a single column.

def nn_inference(layers: int , units: list[int], biases: list[any], weights: list[any],  activation: str, input_vector: any) -> tuple: 
    a = [None] * (layers + 1)
    z = [None] * (layers + 1)

    z[1] = input_vector

    for i in range(2, layers + 1): 
        input = z[i-1]
        curr_w = weights[i]
        curr_b= biases[i]
        output_a = curr_w @ input + curr_b
        output_z = sigmoid_function(output_a) if activation == 'sigmoid' else step_function(output_a)

        a[i] = output_a
        z[i] = output_z
    return (a, z)

def step_function(a):
    return np.where(a >= 0, 1, 0)
    
def sigmoid_function(a: float): 
    return 1 / (1 + np.exp(- a))

# return a (a, z)
# a_values is a list. a_values[i] is the vector (numpy array) of outputs of step 1 (dot products plus biases) of all units in the i-th layer.
# a_values[0] should be None, since in our notation there is no layer 0.
# a_values[1] should be None, because layer 1 is the input layer, and no dot products are computed there.
# z_values is a list. z_values[i] is the vector (numpy array) of the outputs of all units in the i-th layer.
# z_values[0] should be None, since in our notation there is no layer 0.
# z_values[1] should be equal to the input vector, since layer 1 is the input layer.