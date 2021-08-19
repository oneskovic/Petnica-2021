import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nonrecurrent_neural_network import NeuralNetwork
from utility.activation_functions import relu_function, sigmoid_function, tanh_function

def run_test():

    input_size = np.random.randint(1, 10)
    output_size = np.random.randint(1, 10)
    shared_weight = np.random.ranf()

    model = nn.Sequential(nn.Linear(input_size, input_size*output_size), nn.Tanh(), nn.Linear(input_size*output_size, output_size))
    random_data = torch.rand(input_size)

    conns_l1 = torch.from_numpy(np.random.choice([0, 1], (input_size*output_size, input_size)))
    conns_l2 = torch.from_numpy(np.random.choice([0, 1], (output_size, input_size*output_size)))

    with torch.no_grad():
        weights1 = torch.ones_like(model[0].weight)*shared_weight * conns_l1
        model[0].weight = nn.Parameter(weights1)
        model[0].bias = nn.Parameter(torch.ones_like(model[0].bias)*shared_weight)
        weights2 = torch.ones_like(model[2].weight)*shared_weight * conns_l2
        model[2].weight = nn.Parameter(weights2)
        model[2].bias = nn.Parameter(torch.ones_like(model[2].bias)*shared_weight)
        torch_output = model(random_data)

    custom_nn = NeuralNetwork(input_size, output_size)
    output_neurons = custom_nn.output_neurons
    middle_layer = list()
    for input_neuron in custom_nn.input_neurons:
        for output_neuron in custom_nn.output_neurons:
            middle_layer.append(custom_nn.add_neuron(input_neuron, output_neuron, tanh_function))

    for i in range(len(custom_nn.input_neurons)):
        for j in range(len(middle_layer)):
            input_neuron = custom_nn.input_neurons[i]
            middle_neuron = middle_layer[j]
            if conns_l1[j][i] == 1:
                custom_nn.connect_neurons(input_neuron, middle_neuron, 1)
            else:
                custom_nn.disconnect_neurons(input_neuron, middle_neuron)

    for i in range(len(middle_layer)):
        for j in range(len(output_neurons)):
            middle_neuron = middle_layer[i]
            output_neuron = output_neurons[j]
            if conns_l2[j][i] == 1:
                custom_nn.connect_neurons(middle_neuron, output_neuron, 1)
            else:
                custom_nn.disconnect_neurons(middle_neuron, output_neuron)

    custom_nn.clear_network()
    custom_nn.set_shared_weight(shared_weight)
    random_data_np = np.array(random_data)
    custom_nn.set_input(random_data_np)
    custom_nn_output = custom_nn.compute_activations()

    torch_output = np.array(torch_output)
    diff = torch_output - custom_nn_output
    diff = np.abs(diff)
    return diff

diffs = np.zeros(0)
for _ in range(10000):
    diff = run_test()
    print(diff.max())
    diffs = np.append(diffs, diff.max())

print(diffs.max())