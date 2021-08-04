import numpy as np
from numpy import ndarray

class ComputationGraph:

    def __init__(self):
        self.transpose_adjacency_matrix = None
        self.function_list = list()
        self.node_values = None

    def add_node(self, function):
        self.function_list.append(function)
        if self.transpose_adjacency_matrix is None:
            self.node_values = np.zeros(1)
            self.transpose_adjacency_matrix = np.zeros((1, 1), dtype=bool)
        else:
            self.node_values = np.append(self.node_values, 0)
            self.transpose_adjacency_matrix = np.vstack((self.transpose_adjacency_matrix, np.zeros(len(self.node_values) - 1, dtype=bool)))
            self.transpose_adjacency_matrix = np.hstack((self.transpose_adjacency_matrix, np.zeros((len(self.node_values),1), dtype=bool)))

        return len(self.node_values)-1

    def add_edge(self, node1, node2, weight):
        self.transpose_adjacency_matrix[node2][node1] = 1

    def remove_edge(self, node1, node2):
        self.transpose_adjacency_matrix[node2][node1] = 0

    def get_weight_between_nodes(self, node1, node2):
        return 1

    def get_weight_with_index(self, node1, index):
        return 1

    def set_node_value(self, node, value):
        self.node_values[node] = value

    def evaluate(self, weight):
        self.node_values = np.matmul(self.transpose_adjacency_matrix,self.node_values)
        self.node_values = self.node_values * weight
        for i in range(len(self.node_values)):
            self.node_values[i] = self.function_list[i](self.node_values[i])

    def get_node_value(self, node):
        return self.node_values[node]

    def get_node_values(self):
        return self.node_values

    def set_node_values(self, new_node_values: np.ndarray):
        self.node_values = np.copy(new_node_values)
