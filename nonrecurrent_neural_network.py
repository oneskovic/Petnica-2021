import numpy as np
from utility.computation_graph import ComputationGraph
from utility.activation_functions import identity_function, sigmoid_function

class NeuralNetwork:

    def __init__(self, input_size, output_size):
        self.neuron_count = input_size+output_size
        self.input_neurons = np.array(range(0, input_size))
        self.output_neurons = np.array(range(input_size, input_size + output_size))
        self.non_output_neurons = np.array(self.input_neurons)
        self.hidden_neurons = np.zeros(0, dtype=int)
        self.computation_graph = ComputationGraph()
        self.shared_weight = 0.0
        for _ in range(input_size):                                             # Add input neurons
            self.computation_graph.add_node(identity_function)
        for _ in range(output_size):                                            # Add output neurons
            self.computation_graph.add_node(identity_function)

        # for input_neuron in self.input_neurons:                                 # Connect all input neurons to all output neurons
        #     for output_neuron in self.output_neurons:
        #         self.computation_graph.add_edge(input_neuron, output_neuron, 1)

        self.layers = list([self.input_neurons, self.output_neurons])
        self.neuron_layer_map = dict()
        for neuron in self.input_neurons:
            self.neuron_layer_map[neuron] = 0
        for neuron in self.output_neurons:
            self.neuron_layer_map[neuron] = 1

    def __create_new_layer(self, new_neuron, left_layer):
        for layer_index in range(left_layer+1, len(self.layers)):
            for neuron in self.layers[layer_index]:
                self.neuron_layer_map[neuron] += 1
        self.layers.insert(left_layer+1, np.array([new_neuron]))
        self.neuron_layer_map[new_neuron] = left_layer+1

    def add_neuron(self, neuron1, neuron2, activation_function):
        """
        Adds a neuron between neurons with given indices.
        Assumes the neurons 1 and 2 are already connected.

        Returns:
            The index of the newly added neuron.
        """
        old_weight = self.computation_graph.get_weight_between_nodes(neuron1, neuron2)
        new_node = self.computation_graph.add_node(activation_function)
        self.computation_graph.add_edge(neuron1, new_node, old_weight)
        self.computation_graph.add_edge(new_node, neuron2, old_weight)
        self.computation_graph.remove_edge(neuron1, neuron2)

        bias_node = self.computation_graph.add_node(identity_function)
        self.computation_graph.add_edge(bias_node, new_node, 1)

        self.neuron_count += 1
        self.non_output_neurons = np.append(self.non_output_neurons, new_node)
        self.hidden_neurons = np.append(self.hidden_neurons, new_node)

        left_layer = self.neuron_layer_map[neuron1]
        right_layer = self.neuron_layer_map[neuron2]
        if right_layer <= left_layer:
            raise ValueError('Recurrent or side connection attempted.')

        if right_layer - left_layer == 1:
            self.__create_new_layer(new_node, left_layer)
        else:
            self.neuron_layer_map[new_node] = left_layer+1
            self.layers[left_layer+1] = np.append(self.layers[left_layer+1], new_node)

        return new_node

    def connect_neurons(self, neuron1, neuron2, weight):
        left_layer = self.neuron_layer_map[neuron1]
        right_layer = self.neuron_layer_map[neuron2]
        if right_layer <= left_layer:
            raise ValueError('Recurrent or side connection attempted.')

        return self.computation_graph.add_edge(neuron1, neuron2, weight)

    def disconnect_neurons(self, neuron1, neuron2):
        self.computation_graph.remove_edge(neuron1, neuron2)

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
        self.computation_graph.evaluate_with_layers(self.shared_weight, self.layers)
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
        return np.nonzero(self.computation_graph.transpose_adjacency_matrix[:,neuron])

    def get_weight(self, neuron1, index):
        return self.computation_graph.get_weight_with_index(neuron1, index)

    def get_disconnected_neurons(self):
        disconnected = list()
        for neuron1 in np.append(self.input_neurons,self.hidden_neurons):
            for neuron2 in np.append(self.hidden_neurons, self.output_neurons):
                are_connected = self.computation_graph.transpose_adjacency_matrix[neuron2][neuron1]
                if self.neuron_layer_map[neuron1] < self.neuron_layer_map[neuron2] and not are_connected:
                    disconnected.append((neuron1,neuron2))
        return disconnected

    def set_shared_weight(self, weight):
        self.shared_weight = weight