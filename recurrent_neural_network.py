import numpy as np
from computation_graph import ComputationGraph
from activation_functions import identity_function, sigmoid_function

inovation_id_map = dict()


class NeuralNetwork:

    def __init__(self, input_size, output_size, hyperparameters):
        self.input_neurons = range(0, input_size)
        self.output_neurons = range(input_size, input_size + output_size)
        self.computation_graph = ComputationGraph()
        self.hparams = hyperparameters
        for _ in range(input_size):
            self.computation_graph.add_node(identity_function)
        for _ in range(output_size):
            self.computation_graph.add_node(sigmoid_function)

    def add_neuron(self, neuron1, neuron2, activation_function):
        """
        Adds a neuron between neurons with given indices
        Assumes the neurons 1 and 2 are already connected

        Returns:
            The index of the newly added node.
        TODO: Replace fixed weight
        """
        old_weight = self.computation_graph.get_weight_between_nodes(neuron1, neuron2)
        new_node = self.computation_graph.add_node(activation_function)
        self.computation_graph.add_edge(neuron1, new_node, old_weight)
        self.computation_graph.add_edge(new_node, neuron2, old_weight)
        self.computation_graph.remove_edge(neuron1,neuron2)
        return new_node

    def connect_neurons(self, neuron1, neuron2, weight):
        return self.computation_graph.add_edge(neuron1, neuron2, weight)

    def set_input(self, input_data):
        for i in range(len(input_data)):
            self.computation_graph.set_node_value(self.input_neurons[i], input_data[i])

    def compute_activations(self):
        """
        Steps the network a single step through time.
        Returns:
        The output layer of the network.
        """
        self.computation_graph.evaluate()
        output = np.zeros_like(self.output_neurons, dtype=np.float32)
        for i in range(len(output)):
            output[i] = self.computation_graph.get_node_value(self.output_neurons[i])
        #self.__decay_activations()
        return output

    def __decay_activations(self):
        delta = self.hparams['activation_decay']
        values = self.computation_graph.get_node_values()
        values *= (1-delta)
        self.computation_graph.set_node_values(values)