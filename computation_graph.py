import numpy as np
import queue


class ComputationGraph:
    def __init__(self):
        self.adjacency_list = list()
        self.function_list = list()
        self.weights = list()
        self.node_values = list()
        self.node_arguments = list()

    def add_node(self, function):
        self.adjacency_list.append(list())
        self.function_list.append(function)
        self.node_values.append(0)
        self.node_arguments.append(list())
        self.weights.append(list())
        return len(self.node_values)-1

    def add_edge(self, node1, node2, weight):
        """
        Tries to add the edge node1->node2, if the given edge does not create a cycle
        returns True and adds the edge, otherwise returns False.
        """
        self.adjacency_list[node1].append(node2)
        self.weights[node1].append(weight)
        sorted_nodes = self.__topological_sort()
        if len(sorted_nodes) < len(self.node_values):
            del self.adjacency_list[node1][-1]
            del self.weights[node1][-1]
            return False
        return True

    def set_arguments_at_node(self, node, args):
        self.node_arguments[node] = args

    def __topological_sort(self):
        sorted_nodes = list()
        node_count = len(self.adjacency_list)
        in_degree_of_node = np.zeros(node_count)

        for node in range(node_count):
            for neighbor in self.adjacency_list[node]:
                in_degree_of_node[neighbor] += 1

        in_degree_zero = queue.Queue()
        for node in range(node_count):
            if in_degree_of_node[node] == 0:
                in_degree_zero.put(node)

        while not in_degree_zero.empty():
            node = in_degree_zero.get()
            sorted_nodes.append(node)
            for neighbor in self.adjacency_list[node]:
                in_degree_of_node[neighbor] -= 1
                if in_degree_of_node[neighbor] == 0:
                    in_degree_zero.put(neighbor)

        return sorted_nodes

    def clear_arguments(self):
        for node in range(len(self.node_values)):
            self.node_arguments[node] = list()

    def evaluate(self):
        sorted_nodes = self.__topological_sort()
        for node in sorted_nodes:
            self.node_values[node] = self.function_list[node](self.node_arguments[node])
            for i in range(len(self.adjacency_list[node])):
                neighbor = self.adjacency_list[node][i]
                weight = self.weights[node][i]
                self.node_arguments[neighbor].append(self.node_values[node]*weight)


    def get_value_at_node(self, node):
        return self.node_values[node]