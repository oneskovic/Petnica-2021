import numpy as np
import queue
from copy import deepcopy

class ComputationGraph:

    def __init__(self):
        self.adjacency_list = list()
        self.transpose_adj_list = list()
        self.function_list = list()
        self.weights = list()
        self.transpose_weights = list()
        self.node_values = np.zeros(0)

    def add_node(self, function):
        self.adjacency_list.append(list())
        self.transpose_adj_list.append(list())
        self.function_list.append(function)
        self.node_values = np.append(self.node_values, 0)
        self.weights.append(list())
        self.transpose_weights.append(list())
        return len(self.node_values)-1

    def add_edge(self, node1, node2, weight):
        self.adjacency_list[node1].append(node2)
        self.transpose_adj_list[node2].append(node1)
        self.weights[node1].append(weight)
        self.transpose_weights[node2].append(weight)

    def remove_edge(self, node1, node2):
        index_as_neighbor = self.adjacency_list[node1].index(node2)
        del self.adjacency_list[node1][index_as_neighbor]
        del self.weights[node1][index_as_neighbor]

        index_as_neighbor = self.transpose_adj_list[node2].index(node1)
        del self.transpose_adj_list[node2][index_as_neighbor]
        del self.transpose_weights[node2][index_as_neighbor]

    def get_weight_between_nodes(self, node1, node2):
        index_as_neighbor = self.adjacency_list[node1].index(node2)
        return self.weights[node1][index_as_neighbor]

    def set_node_value(self, node, value):
        self.node_values[node] = value

    def evaluate(self):
        new_values = deepcopy(self.node_values)     # Make an array for new node values
        for node in range(len(new_values)):
            args = np.zeros(len(self.transpose_adj_list[node]))
            for i in range(len(args)):
                args[i] = self.node_values[self.transpose_adj_list[node][i]]*self.transpose_weights[node][i]
            if len(args) > 0:
                new_values[node] = self.function_list[node](args)
        self.node_values = new_values               # Replace old node values with new node values

    def get_node_value(self, node):
        return self.node_values[node]

    def get_node_values(self):
        return self.node_values

    def set_node_values(self, new_node_values):
        self.node_values = deepcopy(new_node_values)
