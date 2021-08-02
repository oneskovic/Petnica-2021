import numpy as np
from utility.computation_graph import ComputationGraph
from utility.activation_functions import identity_function, sigmoid_function

class NeuralNetwork:

    def __init__(self, input_size, output_size):
        self.neuron_count = input_size+output_size
        self.input_neurons = np.array(range(0, input_size))
        self.output_neurons = np.array(range(input_size, input_size + output_size))
        self.non_output_neurons = np.array(self.input_neurons)
        self.computation_graph = ComputationGraph()
        for _ in range(input_size):
            self.computation_graph.add_node(identity_function)
        for _ in range(output_size):
            self.computation_graph.add_node(sigmoid_function)

    def add_neuron(self, neuron1, neuron2, activation_function):
        """
        Adds a neuron between neurons with given indices.
        Assumes the neurons 1 and 2 are already connected.

        Returns:
            The index of the newly added neuron.
        TODO: Replace fixed weight
        """
        old_weight = self.computation_graph.get_weight_between_nodes(neuron1, neuron2)
        new_node = self.computation_graph.add_node(activation_function)
        self.computation_graph.add_edge(neuron1, new_node, old_weight)
        self.computation_graph.add_edge(new_node, neuron2, old_weight)
        self.computation_graph.remove_edge(neuron1,neuron2)
        self.neuron_count += 1
        self.non_output_neurons = np.append(self.non_output_neurons, new_node)
        return new_node

    def connect_neurons(self, neuron1, neuron2, weight):
        return self.computation_graph.add_edge(neuron1, neuron2, weight)

    def set_input(self, input_data):
        for i in range(len(input_data)):
            self.computation_graph.set_node_value(self.input_neurons[i], input_data[i])

    def clear_network(self):
        self.computation_graph.set_node_values(np.zeros_like(self.computation_graph.get_node_values()))

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
        return output

    def get_input_neuron_indices(self):
        return self.input_neurons

    def get_output_neuron_indices(self):
        return self.output_neurons

    def get_non_output_neurons(self):
        return self.non_output_neurons

    def get_connected_neurons(self, neuron):
        return self.computation_graph.adjacency_list[neuron]

    def get_weight(self, neuron1, index):
        return self.computation_graph.get_weight_with_index(neuron1, index)
