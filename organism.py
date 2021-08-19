import numpy as np

import nonrecurrent_neural_network
import recurrent_neural_network
import utility.activation_functions
from utility.activation_functions import all_activation_functions
from copy import deepcopy

innovation_id_map = dict()
innovation_counter = 0


class Organism:
    def __init__(self, input_layer_size, output_layer_size, recurrent=False):
        if recurrent:
            self.neural_net = recurrent_neural_network.NeuralNetwork(input_layer_size, output_layer_size)
        else:
            self.neural_net = nonrecurrent_neural_network.NeuralNetwork(input_layer_size, output_layer_size)

        self.recurrent = recurrent
        self.gene_ids = np.array([], dtype=int)
        self.gene_weights = np.random.rand(input_layer_size*output_layer_size)
        self.start_gene_count = 0
        self.fitness = None
        self.rank = None

        # for input_neuron in self.neural_net.get_input_neuron_indices():
        #     for output_neuron in self.neural_net.get_output_neuron_indices():
        #         self.neural_net.connect_neurons(input_neuron, output_neuron, np.random.ranf())
        #         self.gene_ids = np.append(self.gene_ids, self.__get_inovation_id(input_neuron,output_neuron))

    def __get_inovation_id(self, neuron1, neuron2):
        global innovation_id_map, innovation_counter
        if (neuron1, neuron2) not in innovation_id_map:
            innovation_id_map[(neuron1, neuron2)] = innovation_counter
            innovation_counter += 1
        return innovation_id_map[(neuron1, neuron2)]

    def get_distance_from(self, other_organism, gene_coeff, weight_coeff):
        gene_intersection, indices_a, indices_b = np.intersect1d(self.gene_ids, other_organism.gene_ids, return_indices=True)

        different_genes = max(len(self.gene_ids) - len(indices_a), len(other_organism.gene_ids) - len(indices_b))
        weight_difference = np.abs(self.gene_weights[indices_a] - other_organism.gene_weights[indices_b])
        longest_genome = max(len(self.gene_ids), len(other_organism.gene_ids))
        weight_difference = np.mean(weight_difference)
        different_genes = different_genes / (1+longest_genome)
        return gene_coeff * different_genes + weight_coeff * weight_difference

    def __mutate_recurrent(self, mutation_id):
        if mutation_id == 0:
            neuron1 = np.random.choice(self.neural_net.get_non_output_neurons())

            connected_neurons = self.neural_net.get_connected_neurons(neuron1)[0]
            if len(connected_neurons) is 0:
                raise ValueError('Attempted to add neuron to invalid connection')
            neuron2 = np.random.choice(connected_neurons)
            prev_weight = self.neural_net.get_weight(neuron1,neuron2)

            function = np.random.choice(all_activation_functions)
            new_neuron = self.neural_net.add_neuron(neuron1, neuron2, function)

            self.gene_ids = np.append(self.gene_ids,
                  [self.__get_inovation_id(neuron1, new_neuron),self.__get_inovation_id(new_neuron, neuron2)])
            self.gene_weights = np.append(self.gene_weights, [prev_weight, prev_weight])

        elif mutation_id == 1:
            neuron1 = np.random.choice(self.neural_net.non_output_neurons)
            neuron2 = np.random.choice(self.neural_net.non_output_neurons)
            self.neural_net.connect_neurons(neuron1, neuron2, 1)

            self.gene_ids = np.append(self.gene_ids, self.__get_inovation_id(neuron1, neuron2))
            self.gene_weights = np.append(self.gene_weights, 1)

        else:
            neuron1 = np.random.choice(self.neural_net.non_output_neurons)
            self.neural_net.computation_graph.function_list[neuron1] = np.random.choice(all_activation_functions)

    def mutate_nonrecurrent(self, mutation_id, logger=None, generation_number=0):
        if mutation_id == 0:        # Add a neuron into an existing connection
            non_output_neurons = self.neural_net.get_non_output_neurons()
            valid_neurons = [neuron for neuron in non_output_neurons
                             if len(self.neural_net.get_connected_neurons(neuron)[0]) > 0]
            if len(valid_neurons) == 0:
                if logger is None:
                    print('WARN: All neurons disconnected - mutation failed')
                else:
                    logger.log_msg('WARN: All neurons disconnected - mutation failed', generation_number)
                return
            neuron1 = np.random.choice(valid_neurons)

            connected_neurons = self.neural_net.get_connected_neurons(neuron1)[0]
            if len(connected_neurons) is 0:
                raise ValueError('Attempted to add neuron to invalid connection')
            neuron2 = np.random.choice(connected_neurons)
            prev_weight = self.neural_net.get_weight(neuron1,neuron2)

            function = utility.activation_functions.identity_function
            new_neuron = self.neural_net.add_neuron(neuron1, neuron2, function)

            innovation_id1 = self.__get_inovation_id(neuron1, new_neuron)
            innovation_id2 = self.__get_inovation_id(new_neuron, neuron2)

            if innovation_id1 in self.gene_ids or innovation_id2 in self.gene_ids:
                pass
            self.gene_ids = np.append(self.gene_ids,
                  np.array([innovation_id1, innovation_id2]))
            self.gene_weights = np.append(self.gene_weights, [prev_weight, prev_weight])

        elif mutation_id == 1:      # Add a new connection
            disconnected = self.neural_net.get_disconnected_neurons()
            if len(disconnected) == 0:
                if logger is None:
                    print('WARN: All neurons connected - mutation failed')
                else:
                    logger.log_msg('WARN: All neurons connected - mutation failed', generation_number)
                return
            neurons = disconnected[np.random.randint(0, len(disconnected))]

            self.neural_net.connect_neurons(neurons[0], neurons[1], 1)

            innovation_id = self.__get_inovation_id(neurons[0], neurons[1])
            if innovation_id in self.gene_ids:
                pass
            self.gene_ids = np.append(self.gene_ids, innovation_id)
            self.gene_weights = np.append(self.gene_weights, 1)

        else:                       # Change activation function
            if len(self.neural_net.hidden_neurons) == 0:
                if logger is None:
                    print('WARN: No hidden neurons, could not mutate activations - mutation failed')
                else:
                    logger.log_msg('WARN: No hidden neurons, could not mutate activations - mutation failed', generation_number)
                return
            neuron1 = np.random.choice(self.neural_net.hidden_neurons)
            self.neural_net.computation_graph.function_list[neuron1] = np.random.choice(all_activation_functions)

    def mutate(self, hparams, logger=None, generation_number=0):
        """
        Insert a new neuron into an existing connection path (with probability given in hparams).
        Add a new directed connection (with probability given in hparams).
        Change actiavation function (with probability given in hparams).
        Assign species to None.
        """
        mutation_id = np.random.choice([0,1,2], p=[hparams['prob_mutate_add_neuron'],
                                        hparams['prob_mutate_add_connection'],
                                        hparams['prob_mutate_change_activation']])

        if self.recurrent:
            self.__mutate_recurrent(mutation_id)
        else:
            self.mutate_nonrecurrent(mutation_id, logger, generation_number)

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

        child.assign_species(None)
        return child


