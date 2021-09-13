from typing import List, Any, Union

import numpy as np
import pickle
from organism import Organism
from utility.nsga_sort import nsga_sort
from copy import deepcopy
from utility.logger import Logger
import operator
import math
from recurrent_neural_network import NeuralNetwork
from numpy.random import default_rng


class GeneticAlgorithm:
    population: List[Organism]
    logger: Logger

    def __init__(self, hparams, problem_params, logger):
        self.generation_number = 0
        self.hparams = hparams
        self.logger = logger
        self.problem_params = problem_params
        if 'random_seed' in self.hparams:
            self.rng = np.random.default_rng(self.hparams['random_seed'])
            self.problem_params['evaluator'].eval_env.seed(self.hparams['random_seed'])
        else:
            self.rng = np.random.default_rng()
        self.__init_population(hparams['population_size'])

    def __init_population(self, population_size):
        input_layer_size = self.problem_params['input_layer_size']
        output_layer_size = self.problem_params['output_layer_size']
        self.population = [Organism(input_layer_size, output_layer_size, self.hparams['recurrent_nets'], self.rng) for _ in range(population_size)]

        for organism in self.population:
            nn = organism.neural_net
            if organism.recurrent:
                connection_cnt = (len(nn.input_neurons) + len(nn.state_input_neurons))*(len(nn.output_neurons) + len(nn.state_output_neurons))
                connection_cnt *= self.hparams['init_connection_fraction']
                connection_cnt = int(connection_cnt)
                for _ in range(connection_cnt):
                    if self.rng.random() <= self.hparams['init_connection_prob']:
                        organism.mutate_nonrecurrent(1, self.logger, self.generation_number)
            else:
                for input_neuron in nn.input_neurons:
                    for output_neuron in nn.output_neurons:
                        if self.rng.random() <= 0.5:
                            organism.mutate_nonrecurrent(1, self.logger, self.generation_number)

            # Make sure there is at least one connection
            organism.mutate_nonrecurrent(1, self.logger, self.generation_number)
            # Add a single hidden neuron
            organism.mutate_nonrecurrent(0, self.logger, self.generation_number)

        self.__evaluate_population(self.population)

        all_scores = np.array([organism.fitness for organism in self.population])
        if self.rng.random() < self.hparams['prob_multiobjective']:
            all_scores = all_scores[:, [0, 1]]
        else:
            all_scores = all_scores[:, [0, 2]]

        organism_index_rank = nsga_sort(all_scores)
        for i in range(len(organism_index_rank)):
            self.population[i].rank = organism_index_rank[i]

        self.population.sort(key=operator.attrgetter('rank'))

    def __create_offspring(self, parent_organisms, offspring_count):
        """
        Creates offspring for the given species.
        Assumes the organisms in self.population are already sorted.
        """
        offspring = list()

        # Create a tournament for the first and second parent of size given in hparams
        parent1_tournament = self.rng.choice(parent_organisms, size=(offspring_count, self.hparams['tournament_size']))
        parent2_tournament = self.rng.choice(parent_organisms, size=(offspring_count, self.hparams['tournament_size']))
        # Organisms are already sorted so comparing by index is enough
        parents1 = parent1_tournament.min(axis=1)
        parents2 = parent2_tournament.min(axis=1)
        for i in range(offspring_count):
            if self.rng.random() < self.hparams['prob_mutate']:
                organism_index = min(parents1[i], parents2[i])
                new_organism = deepcopy(self.population[organism_index])

            else:
                parent_indices = [parents1[i], parents2[i]]
                parent1 = self.population[parent_indices[0]]
                parent2 = self.population[parent_indices[1]]
                new_organism = parent1.crossover(parent2)

            new_organism.mutate(self.hparams, self.logger, self.generation_number)
            offspring.append(new_organism)
        return offspring

    def __evaluate_population(self, population):
        if self.hparams['thread_count'] > 1:
            self.problem_params['evaluator'].evaluate_population_parallel(population)
        else:
            self.problem_params['evaluator'].evaluate_population_serial(population)

    def __get_stats(self):
        scores = [self.population[i].fitness for i in range(len(self.population))]
        genomes = [self.population[i].gene_ids for i in range(len(self.population))]
        best_organism = deepcopy(self.population[0])

        self.logger.log_value('scores', self.generation_number, scores)
        self.logger.log_value('best_organism', self.generation_number, best_organism)
        self.logger.log_value('hparams', 0, deepcopy(self.hparams))
        self.logger.log_value('problem_params', 0, deepcopy(self.problem_params))

        if self.logger.detailed:
            self.logger.log_value('genomes', self.generation_number, genomes)
            self.logger.log_value('organisms', self.generation_number, deepcopy(self.population))

    def step_generation(self):
        self.__get_stats()

        debug_arr = np.array([o.rank for o in self.population], dtype=int)
        if not np.all(np.diff(np.abs(debug_arr)) >= 0):
            print(debug_arr)
            print('WARN: Population not sorted by rank!')

        remove_count = int(self.hparams['dieoff_fraction']*len(self.population))
        elite_count = int(self.hparams['elite_fraction']*len(self.population))

        if remove_count > 0:
            del self.population[-remove_count:]

        elite_organisms = list([])
        elite_count = min(elite_count, len(self.population))
        if elite_count > 0:
            elite_organisms = self.population[:elite_count]

        new_organisms = self.__create_offspring(np.arange(0, len(self.population)),
                                                self.hparams['population_size']-elite_count)

        self.population = elite_organisms + new_organisms           # Join the elite population with the new population
        self.__evaluate_population(self.population)

        # all_scores[i][0] = average reward
        # all_scores[i][1] = max average reward
        # all_scores[i][2] = number of connections (genes)
        all_scores = np.array([organism.fitness for organism in self.population])
        if self.rng.random() < self.hparams['prob_multiobjective']:
            all_scores = all_scores[:, [0, 1]]
        else:
            all_scores = all_scores[:, [0, 2]]

        organism_index_rank = nsga_sort(all_scores)
        for i in range(len(organism_index_rank)):
            self.population[i].rank = organism_index_rank[i]

        self.population.sort(key=operator.attrgetter('rank'))
        if len(self.population) != self.hparams['population_size']:
            raise ValueError('Incorrect population size!')
        self.generation_number += 1
