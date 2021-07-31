import numpy as np
from computation_graph import ComputationGraph
from activation_functions import identity_function, sigmoid_function


class NeuralNetwork:
    def __init__(self, input_size, output_size):
        self.input_neurons = range(0, input_size)
        self.output_neurons = range(input_size, input_size + output_size)
        self.computation_graph = ComputationGraph()
        for _ in range(input_size):
            self.computation_graph.add_node(identity_function)
        for _ in range(output_size):
            self.computation_graph.add_node(sigmoid_function)

    def add_neuron(self, activation_function):
        return self.computation_graph.add_node(activation_function)

    def connect_neurons(self, neuron1, neuron2, weight):
        return self.computation_graph.add_edge(neuron1, neuron2, weight)

    def set_input(self, input_data):
        self.computation_graph.clear_arguments()
        for i in range(len(input_data)):
            self.computation_graph.set_arguments_at_node(self.input_neurons[i], [input_data[i]])

    def feed_forward(self):
        self.computation_graph.evaluate()
        output = np.zeros_like(self.output_neurons, dtype=np.float32)
        for i in range(len(output)):
            output[i] = self.computation_graph.get_value_at_node(self.output_neurons[i])
        return output
