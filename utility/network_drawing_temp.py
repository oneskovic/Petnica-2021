import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
from recurrent_neural_network import NeuralNetwork
from utility.activation_functions import sigmoid_function

def draw_neural_net(nn: NeuralNetwork):
    G = nx.from_numpy_matrix(nn.computation_graph.transpose_adjacency_matrix.T, create_using=nx.MultiDiGraph())
    layer_cnt = len(nn.layers)
    node_pos = dict()
    for i in range(layer_cnt):
        for j in range(len(nn.layers[i])):
            node = nn.layers[i][j]
            node_pos[node] = (i, j)

    color_map = []
    for node in range(nn.computation_graph.transpose_adjacency_matrix.shape[0]):
        if node in nn.input_neurons:
            color_map.append('yellow')
        elif node in nn.output_neurons:
            color_map.append('red')
        elif node in nn.state_input_neurons or node in nn.state_output_neurons:
            color_map.append('green')
        else:
            color_map.append('blue')

    for node in range(nn.computation_graph.transpose_adjacency_matrix.shape[0]):
        if len(nn.get_connected_neurons(node)) == 0:
            G.remove_node(node)

    # node_pos = nx.spring_layout(G)
    nx.draw(G, node_color=color_map, with_labels=True, pos=node_pos, font_color='white')
    plt.show()
# fig = plt.figure()
# ax1 = fig.add_subplot(1,1,1)
#
#
# def get_node_positions(input_size, output_size, neuron_count):
#     positions = dict()
#     for i in range(input_size):
#         positions[i] = (0,i)
#     for i in range(output_size):
#         positions[i+input_size] = (2,i)
#     for i in range(neuron_count - input_size - output_size):
#         positions[input_size+output_size+i] = (1,i)
#     return positions
#
# input_size = 4
# output_size = 2
# hparams = {
#     'activation_decay': 0.5
# }
# net = NeuralNetwork(input_size,output_size,hparams)
#
# input_neuron_indices = np.zeros(input_size)
# output_neuron_indices = np.zeros(output_size)
# G = nx.DiGraph()
# for i in range(input_size):
#     G.add_node(i,color='blue')
#     input_neuron_indices[i] = i
#
# for i in range(output_size):
#     G.add_node(input_size+i)
#     output_neuron_indices[i] = input_size+i
#
# for i in range(input_size):
#     for j in range(output_size):
#         net.connect_neurons(i, input_size+j, np.random.ranf())
#         G.add_edge(i,input_size+j)
#
# for _ in range(10):
#     neuron1 = np.random.randint(input_size)
#     neuron2 = np.random.choice(net.computation_graph.adjacency_list[neuron1])
#
#     new_neuron_index = net.add_neuron(neuron1, neuron2, sigmoid_function)
#     G.add_edge(neuron1, new_neuron_index)
#     G.add_edge(new_neuron_index, neuron2)
#
# # colors = [node[1]['color'] for node in G.nodes(data=True)]
# net.set_input([0.2,0.1,0.5,0.3])
# node_pos = get_node_positions(input_size,output_size,input_size+output_size+10)
#
# def animate(i):
#     ax1.clear()
#     net.compute_activations()
#     print(net.computation_graph.get_node_values())
#     nx.draw(G, node_color=net.computation_graph.get_node_values(), with_labels=True,
#             font_color='white', cmap = plt.cm.Blues, vmin=0, vmax=1, pos=node_pos)
#     plt.show()
#
# ani = animation.FuncAnimation(fig, animate, interval=1000)
# plt.show()
