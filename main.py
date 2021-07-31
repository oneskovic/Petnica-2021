from computation_graph import ComputationGraph
import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
import networkx as nx

imagefile = 'C:/Users/Ognjen/Desktop/Petnica/Projekat2021/train-images.idx3-ubyte'
imagearray = idx2numpy.convert_from_file(imagefile)

plt.imshow(imagearray[4], cmap=plt.cm.binary)
plt.show()

labels = idx2numpy.convert_from_file('C:/Users/Ognjen/Desktop/Petnica/Projekat2021/train-labels.idx1-ubyte')

def identity_function(x):
    return x


def sigmoid_function(x):
    x = np.array(x, dtype=np.float32)
    x = x.sum()
    return 1 / (1 + np.exp(-x))

from neural_network import NeuralNetwork

input_size = len(imagearray[0].flatten())
net = NeuralNetwork(input_size, 10)

G = nx.Graph()
for i in range(input_size):
    G.add_node(i,color='red')
for i in range(10):
    G.add_node(input_size+i,color='green')

for i in range(200):
    net.add_neuron(sigmoid_function)
    G.add_node(input_size+10+i, color='blue')

for _ in range(2000):
    neuron1 = np.random.randint(input_size+10+200)
    neuron2 = np.random.randint(input_size+10+200)
    G.add_edge(neuron1,neuron2)
    net.connect_neurons(neuron1,neuron2,np.random.ranf())

colors = [node[1]['color'] for node in G.nodes(data=True)]
nx.draw(G, node_color=colors, with_labels=True, font_color='white')
plt.show()

net.set_input(imagearray[0].flatten())
print(net.feed_forward())
net.set_input(imagearray[1].flatten())
print(net.feed_forward())
net.set_input(imagearray[2].flatten())
print(net.feed_forward())
net.set_input(imagearray[3].flatten())
print(net.feed_forward())