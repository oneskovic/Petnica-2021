import numpy as np
from recurrent_neural_network import NeuralNetwork
from activation_functions import all_activation_functions
from copy import deepcopy

innovation_id_map = dict()
innovation_counter = 0


class Organism:
    def __init__(self, input_layer_size, output_layer_size):
        self.neural_net = NeuralNetwork(input_layer_size, output_layer_size)
        self.gene_ids = np.array(np.arange(input_layer_size * output_layer_size))
        self.gene_weights = np.random.rand(input_layer_size*output_layer_size)
        self.start_gene_count = input_layer_size * output_layer_size
        self.species_id = None
        self.fitness = None
        self.is_seed_organism = False

        global innovation_id_map, innovation_counter
        if innovation_counter == 0:
            for input_neuron in self.neural_net.get_input_neuron_indices():
                for output_neuron in self.neural_net.get_output_neuron_indices():
                    innovation_id_map[(input_neuron, output_neuron)] = innovation_counter
                    innovation_counter += 1

        for input_neuron in self.neural_net.get_input_neuron_indices():
            for output_neuron in self.neural_net.get_output_neuron_indices():
                self.neural_net.connect_neurons(input_neuron, output_neuron, np.random.ranf())

    def __get_inovation_id(self, neuron1, neuron2):
        global  innovation_id_map, innovation_counter
        if (neuron1, neuron2) not in innovation_id_map:
            innovation_id_map[(neuron1,neuron2)] = innovation_counter
            innovation_counter += 1
        return innovation_id_map[(neuron1,neuron2)]

    def assign_species(self, species_id):
        self.species_id = species_id

    def assign_fitness(self, fitness):
        self.fitness = fitness

    def assign_seed(self):
        self.is_seed_organism = True

    def get_distance_from(self, other_organism, gene_coeff, weight_coeff):
        gene_intersection, indices_a, indices_b = np.intersect1d(self.gene_ids, other_organism.gene_ids, return_indices=True)
        different_genes = len(self.gene_ids) - len(indices_a) + len(other_organism.gene_ids) - len(indices_b)
        weight_difference = np.abs(self.gene_weights[indices_a] - other_organism.gene_weights[indices_b])
        longest_genome = max(len(self.gene_ids), len(other_organism.gene_ids)) - self.start_gene_count
        weight_difference = np.mean(weight_difference)
        different_genes = different_genes / (1+longest_genome)
        return gene_coeff * different_genes + weight_coeff * weight_difference

    def mutate(self):
        if np.random.ranf() < 0.5:
            neuron1 = np.random.choice(self.neural_net.get_non_output_neurons())
            if not self.neural_net.get_connected_neurons(neuron1):
                print('au buraz')

            connected_neurons = self.neural_net.get_connected_neurons(neuron1)
            neuron2_index = np.random.randint(len(connected_neurons))
            neuron2 = connected_neurons[neuron2_index]
            prev_weight = self.neural_net.get_weight(neuron1,neuron2_index)

            function = np.random.choice(all_activation_functions)
            new_neuron = self.neural_net.add_neuron(neuron1, neuron2, function)

            self.gene_ids = np.append(self.gene_ids,
                  [self.__get_inovation_id(neuron1, new_neuron),self.__get_inovation_id(new_neuron, neuron2)])
            self.gene_weights = np.append(self.gene_weights, [prev_weight, prev_weight])

        else:
            neuron1 = np.random.randint(self.neural_net.neuron_count)
            neuron2 = np.random.randint(self.neural_net.neuron_count)
            weight = np.random.ranf()
            self.neural_net.connect_neurons(neuron1, neuron2, weight)

            self.gene_ids = np.append(self.gene_ids, self.__get_inovation_id(neuron1, neuron2))
            self.gene_weights = np.append(self.gene_weights, weight)


    def crossover(self, other_parent):
        fittest_parent = self
        less_fit_parent = other_parent
        # Order the parents if possible
        if (other_parent.fitness > self.fitness).all():
            fittest_parent, less_fit_parent = other_parent, self
        elif (self.fitness > other_parent.fitness).all():
            fittest_parent, less_fit_parent = self, other_parent
        else:   # If ordering is not possible choose randomly which parent is more fit
            if np.random.ranf() < 0.5:
                fittest_parent, less_fit_parent = other_parent, self


        child = deepcopy(fittest_parent)
        matching, fitter_intersect_ind, less_fit_intersect_ind = \
            np.intersect1d(fittest_parent.gene_ids, less_fit_parent.gene_ids , return_indices=True)

        less_fit_prob = 0.5
        # Boolean array (0 = should not take weight from less fit organism, 1 = should take)
        take_less_fit = np.random.rand(len(matching)) < less_fit_prob
        try:
            child.gene_weights[fitter_intersect_ind[take_less_fit]] = less_fit_parent.gene_weights[less_fit_intersect_ind[take_less_fit]]
        except:
            print('au buraz')

        return child

