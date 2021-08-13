from typing import List, Any, Union

import numpy as np
from organism import Organism
from utility.nsga_sort import nsga_sort
from copy import deepcopy
import progressbar
from utility.logger import Logger
import operator
import math
from numpy.random import default_rng


class GeneticAlgorithm:
    population: List[Organism]
    logger: Logger

    def __init__(self, hparams, problem_params, logger):
        self.generation_number = 0
        self.hparams = hparams
        self.logger = logger
        self.problem_params = problem_params
        self.__init_population(hparams['population_size'])

    def __init_population(self, population_size):
        input_layer_size = self.problem_params['input_layer_size']
        output_layer_size = self.problem_params['output_layer_size']
        self.population = [Organism(input_layer_size, output_layer_size) for _ in range(population_size)]
        print('Generating starting population...')
        #self.__create_good_starting_pop()
        self.population = self.__generate_random_pop(self.hparams['population_size'])

        self.__evaluate_population(self.population)

        all_scores = np.array([organism.fitness for organism in self.population])

        if np.random.ranf() < self.hparams['prob_multiobjective']:
            organism_index_rank = nsga_sort(all_scores)  # Get positions of organisms in the sorted array
            for i in range(len(organism_index_rank)):
                self.population[i].rank = organism_index_rank[i]
            self.multi_objective_sort = True
        else:
            argsort = np.argsort(-all_scores[:, 0])
            for i in range(len(argsort)):
                self.population[argsort[i]].rank = i
            self.multi_objective_sort = False
        self.population.sort(key=operator.attrgetter('rank'))

    def __generate_random_pop(self, count):
        input_layer_size = self.problem_params['input_layer_size']
        output_layer_size = self.problem_params['output_layer_size']
        population = [Organism(input_layer_size,output_layer_size) for _ in range(count)]
        for i in range(len(population)):
            for _ in range(20):
                population[i].mutate(self.hparams)
        return population

    def __create_good_starting_pop(self):
        good_generated = 0
        bar = progressbar.ProgressBar(max_value=len(self.population))
        while good_generated < len(self.population):
            random_pop = self.__generate_random_pop(max(len(self.population)-good_generated,50))
            self.__evaluate_population(random_pop)
            scores = [random_pop[i].fitness for i in range(len(random_pop))]

            # score_buckets = np.linspace(-100, -200, 5)
            # bucket_counts = [25, 250, 450, 250, 25]
            score_buckets = [-400]
            bucket_counts = [50]
            bucket_remaining_cnt = bucket_counts
            for i in range(len(random_pop)):
                for bucket_index in range(len(score_buckets)):
                    if scores[i][0] >= score_buckets[bucket_index]:
                        if bucket_remaining_cnt[bucket_index] > 0 and good_generated < len(self.population):
                            self.population[good_generated] = random_pop[i]
                            good_generated += 1
                            bucket_remaining_cnt[bucket_index] -= 1
                            bar.update(good_generated)
                        break

    def __create_offspring(self, parent_organisms, offspring_count):
        """
        Creates offspring for the given species.
        Assumes the organisms in self.population are already sorted.
        """
        offspring = list()

        # Create a tournament for the first and second parent of size given in hparams
        parent1_tournament = np.random.choice(parent_organisms, size=(offspring_count, self.hparams['tournament_size']))
        parent2_tournament = np.random.choice(parent_organisms, size=(offspring_count, self.hparams['tournament_size']))
        # Organisms are already sorted so comparing by index is enough
        parents1 = parent1_tournament.min(axis=1)
        parents2 = parent2_tournament.min(axis=1)
        for i in range(offspring_count):
            if np.random.ranf() < self.hparams['prob_mutate']:
                organism_index = min(parents1[i], parents2[i])
                new_organism = deepcopy(self.population[organism_index])

            else:
                parent_indices = [parents1[i], parents2[i]]
                parent1 = self.population[parent_indices[0]]
                parent2 = self.population[parent_indices[1]]
                new_organism = parent1.crossover(parent2)

            # for _ in range(max(5, math.ceil(self.hparams['mutate_amount'] * len(new_organism.gene_ids)))):
            new_organism.mutate(self.hparams)
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
        self.logger.log_value('genomes', self.generation_number, genomes)
        self.logger.log_value('best_organism', self.generation_number, best_organism)
        self.logger.log_value('hparams', 0, deepcopy(self.hparams))
        self.logger.log_value('problem_params', 0, deepcopy(self.problem_params))

    def step_generation(self):
        self.__get_stats()

        debug_arr = np.array([o.fitness[0] for o in self.population])
        if not np.all(np.diff(debug_arr) <= 0) and not self.multi_objective_sort:
            print(debug_arr)
            print('WARN: Population not sorted!')
        debug_arr = np.array([o.rank for o in self.population], dtype=int)
        if not np.all(np.diff(np.abs(debug_arr)) >= 0):
            print(debug_arr)
            print('WARN: Population not sorted by rank!')

        new_organisms = self.__create_offspring(np.arange(0, len(self.population)), self.hparams['population_size'])
        self.__evaluate_population(new_organisms)

        self.population = self.population + new_organisms           # Join the old population with the new population
        all_scores = np.array([organism.fitness for organism in self.population])

        if np.random.ranf() < self.hparams['prob_multiobjective']:
            organism_index_rank = nsga_sort(all_scores)             # Get positions of organisms in the sorted array
            for i in range(len(organism_index_rank)):
                self.population[i].rank = organism_index_rank[i]
            self.multi_objective_sort = True
        else:
            argsort = np.argsort(-all_scores[:, 0])
            organism_index_rank = np.zeros(len(argsort), dtype=int)
            for i in range(len(argsort)):
                self.population[argsort[i]].rank = i
                organism_index_rank[argsort[i]] = i
            self.multi_objective_sort = False

        self.population.sort(key=operator.attrgetter('rank'))
        ranks = np.array(np.arange(1, len(self.population)+1), dtype=np.float64)
        ranks -= max(ranks)
        ranks = np.abs(ranks)
        ranks += 1
        ranks /= ranks.sum()
        chosen = default_rng().choice(self.population, self.hparams['population_size'], replace=False, p=ranks)

        self.population = list(chosen)
        self.population.sort(key=operator.attrgetter('rank'))
        self.generation_number += 1
