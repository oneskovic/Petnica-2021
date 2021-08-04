from typing import List, Any, Union

import numpy as np
from organism import Organism
from utility.nsga_sort import nsga_sort
from copy import deepcopy
import progressbar
from utility.logger import Logger
import operator


class GeneticAlgorithm:
    species_max_score: List[int]
    species_last_improvement: List[int]
    population: List[Organism]
    all_species: List[List[int]]
    logger: Logger

    def __init__(self, hparams, problem_params, logger):
        self.generation_number = 0
        self.hparams = hparams
        self.logger = logger
        self.all_species = list()
        self.species_seed_organism = list()
        self.species_last_improvement = list()
        self.species_max_score = list()
        self.problem_params = problem_params
        self.__init_population(hparams['population_size'])
        self.evaluator_list = [deepcopy(self.problem_params['evaluator']) for _ in range(self.hparams['thread_count'])]

    def __init_population(self, population_size):
        input_layer_size = self.problem_params['input_layer_size']
        output_layer_size = self.problem_params['output_layer_size']
        self.population = [Organism(input_layer_size, output_layer_size) for _ in range(population_size)]
        for i in range(population_size):
            for _ in range(20):
                self.population[i].mutate(self.hparams)

        self.__evaluate_population(self.population)

        all_scores = np.array([organism.fitness for organism in self.population])
        organism_index_rank = nsga_sort(all_scores)
        for i in range(len(organism_index_rank)):
            self.population[i].rank = organism_index_rank[i]
        self.__speciate_population()

    def __assign_species(self, organism_index, population, species_list, seed_organism_list, species_last_improvement, species_max_score):
        if population[organism_index].species_id is None:
            was_assigned_species = False
            for species_index in range(len(species_list)):
                dist = population[organism_index].get_distance_from(population[seed_organism_list[species_index]],
                                                                         self.hparams['gene_coeff'], self.hparams['weight_coeff'])
                if dist <= self.hparams['species_threshold']:
                    species_list[species_index].append(organism_index)
                    population[organism_index].assign_species(species_index)
                    if population[organism_index].rank > species_max_score[species_index]:
                        species_max_score[species_index] = population[organism_index].rank
                        species_last_improvement[species_index] = self.generation_number
                    was_assigned_species = True
                    break
            if not was_assigned_species:
                species_list.append(list([organism_index]))
                seed_organism_list.append(organism_index)
                species_max_score.append(organism_index)
                species_last_improvement.append(self.generation_number)
                population[organism_index].assign_species(len(population)-1)
                population[organism_index].assign_seed()

    def best_int_split(self, ratio, total):
        """
        Divides a total into integer shares that best reflects ratio
        # Taken from https://github.com/google/brain-tokyo-workshop/blob/6d0a262171cca7e2e08f901981880e5247b4d677/WANNRelease/prettyNeatWann/utils/utils.py#L57
        Args:
            ratio      - [1 X N ] - Percentage in each pile
            total      - [int   ] - Integer total to split

        Returns:
            intSplit   - [1 x N ] - Number in each pile
        """
        # Handle poorly defined ratio
        if sum(ratio) is not 1:
            ratio = np.asarray(ratio) / sum(ratio)

        # Get share in real and integer values
        floatSplit = np.multiply(ratio, total)
        intSplit = np.floor(floatSplit)
        remainder = int(total - sum(intSplit))

        # Rank piles by most cheated by rounding
        deserving = np.argsort(-(floatSplit - intSplit), axis=0)

        # Distribute remained to most deserving
        intSplit[deserving[:remainder]] = intSplit[deserving[:remainder]] + 1
        return np.array(intSplit, dtype=np.int)

    def __create_offspring(self, species, offspring_count):
        """
        Creates offspring for the given species.
        Assumes the organisms in self.population are already sorted.
        """
        offspring = list()

        # Create a tournament for the first and second parent of size given in hparams
        parent1_torunament = np.random.choice(species, size=(offspring_count, self.hparams['tournament_size']))
        parent2_tournament = np.random.choice(species, size=(offspring_count, self.hparams['tournament_size']))
        # Organisms are already sorted so comparing by index is enough
        parents1 = parent1_torunament.max(axis=1)
        parents2 = parent2_tournament.max(axis=1)
        for i in range(offspring_count):
            if np.random.ranf() < self.hparams['prob_mutate']:
                organism_index = parents1[i]
                new_organism = deepcopy(self.population[organism_index])
                new_organism.mutate(self.hparams)
                offspring.append(new_organism)
            else:
                parent_indices = [parents1[i], parents2[i]]
                parent1 = self.population[parent_indices[0]]
                parent2 = self.population[parent_indices[1]]
                offspring.append(parent1.crossover(parent2))
        return offspring

    def __try_speciate(self):

        species_cpy = [list() for _ in range(len(self.all_species))]
        population_cpy = deepcopy(self.population)
        seed_organism_cpy = deepcopy(self.species_seed_organism)
        species_last_improvement_cpy = deepcopy(self.species_last_improvement)
        species_max_score_cpy = deepcopy(self.species_max_score)

        for organism in population_cpy:
            organism.assign_species(None)

        for organism_index in range(len(population_cpy)):
            self.__assign_species(organism_index, population_cpy, species_cpy, seed_organism_cpy, species_last_improvement_cpy, species_max_score_cpy)
        total_nonempty_species = len([species for species in species_cpy if len(species) > 0])

        return total_nonempty_species

    def __speciate_population(self):
        """
        Clears the current species and assigns species to all organisms in self.population
        so that the upper bound on population size (given in hparams) is respected.
        Assumes the organisms in self.population are sorted.
        """
        for organism in self.population:
            organism.assign_species(None)
        left = 1.0
        right = 0.0
        prev_max = 0

        while left - right > self.hparams['species_threshold_precision']:
            mid = (left+right)/2.0
            self.hparams['species_threshold'] = mid
            total_nonempty_species = self.__try_speciate()
            if prev_max <= total_nonempty_species <= self.hparams['species_count']:
                left = mid
                prev_max = total_nonempty_species
            else:
                right = mid

        self.hparams['species_threshold'] = left
        self.all_species = [list() for _ in range(len(self.all_species))]
        for organism_index in range(len(self.population)):
            self.__assign_species(organism_index, self.population, self.all_species, self.species_seed_organism, self.species_last_improvement, self.species_max_score)


    def __assign_offspring(self, all_species):
        """
        Assigns a number of offspring to each species based on fitnesses of organisms inside the species.
        Assumes the organisms have already been evaluated (ie. that the organism.fitness property was calculated)
        Args:
            all_species: list(list(int)) - indices of organisms in each species
        Returns:
            offspring_count: np.array - offspring count for each species
        """
        avg_scores = np.zeros(len(all_species))
        for species_index in range(len(all_species)):
            total_score = 0.0
            if len(all_species[species_index]) > 0:
                for organism_index in range(len(all_species[species_index])):
                    total_score += np.sum(self.population[organism_index].fitness)
                avg_scores[species_index] = total_score / len(all_species[species_index])

        offspring_count = self.best_int_split(avg_scores, self.hparams['population_size'])
        offspring_count_start = deepcopy(offspring_count)

        nonstagnating_species = list()
        children_to_distribute = 0
        for i in range(len(self.all_species)):
            time_since_improvement = self.generation_number - self.species_last_improvement[i]
            if time_since_improvement > self.hparams['stagnation_start']:
                taken_children = min(offspring_count[i] * 0.08*time_since_improvement**2, offspring_count[i])
                children_to_distribute += taken_children
                offspring_count[i] -= taken_children
            else:
                nonstagnating_species.append(i)
        if len(nonstagnating_species) > 0 and children_to_distribute > 0:
            fraction = 1.0/len(nonstagnating_species)
            split = self.best_int_split([fraction]*len(nonstagnating_species),children_to_distribute)
            for i in range(len(split)):
                spec_index = nonstagnating_species[i]
                offspring_count[spec_index] += split[i]
        else:
            offspring_count = offspring_count_start

        return offspring_count

    def __eval_organism(self, population, population_fitnesses, index, progressbar, mutex, free_evaluators):
        evaluator_index = None
        mutex.acquire()
        evaluator_index = free_evaluators.pop()
        #print(f'Start eval {evaluator_index}')
        mutex.release()

        population[index].assign_fitness(self.evaluator_list[evaluator_index].evaluate_organism(population[index]))
        population_fitnesses[index] = population[index].fitness

        mutex.acquire()
        progressbar.update(progressbar.value + 1)
        free_evaluators.append(evaluator_index)
        #print(f'Finish eval {evaluator_index}')
        mutex.release()

    def __evaluate_population(self, population):
        print('Evaluating population...')
        population_fitnesses = np.zeros((len(population), self.problem_params['evaluator'].get_objective_count()))
        # bar = progressbar.ProgressBar(max_value=len(population))
        # mutex = Lock()
        # pool = ThreadPool(self.hparams['thread_count'])
        #
        # free_evaluators = list(range(self.hparams['thread_count']))
        # pool.starmap(self.__eval_organism,zip(itertools.repeat(population),
        #             itertools.repeat(population_fitnesses), range(len(population)),itertools.repeat(bar),itertools.repeat(mutex),
        #                                       itertools.repeat(free_evaluators)))
        # pool.close()
        # pool.join()
        population_fitnesses = np.zeros((len(population), self.problem_params['evaluator'].get_objective_count()))
        for i in progressbar.progressbar(range(len(population))):
            population[i].assign_fitness(self.problem_params['evaluator'].evaluate_organism(population[i]))
            population_fitnesses[i] = population[i].fitness
        print(' ')

        return population_fitnesses

    def get_stats(self):
        nonempty_species = [species for species in self.all_species if len(species) > 0]
        species_sizes = np.array([len(species) for species in nonempty_species])
        species_scores = [list()] * len(nonempty_species)
        for nonempty_index in range(len(nonempty_species)):
            for organism_index in nonempty_species[nonempty_index]:
                species_scores[nonempty_index].append(self.population[organism_index].fitness)

        for i in range(len(species_scores)):
            species_scores[i] = np.array(species_scores[i], dtype=np.float32)

        species_genomes = [list() for _ in range(len(nonempty_species))]
        for nonempty_index in range(len(nonempty_species)):
            for organism_index in nonempty_species[nonempty_index]:
                species_genomes[nonempty_index].append(self.population[organism_index].gene_ids)

        self.logger.log_value('nonempty_species', self.generation_number, nonempty_species)
        self.logger.log_value('species_sizes', self.generation_number, species_sizes)
        self.logger.log_value('species_scores', self.generation_number, species_scores)
        self.logger.log_value('species_genomes', self.generation_number, species_genomes)

    def __remove_empty_species(self):
        new_all_species = list()
        new_seed_organisms = list()
        species_last_improvement = list()
        species_max_score = list()

        for i in range(len(self.all_species)):
            if len(self.all_species[i]) > 0:
                new_all_species.append(self.all_species[i])
                new_seed_organisms.append(self.species_seed_organism[i])
                species_last_improvement.append(self.species_last_improvement[i])
                species_max_score.append(self.species_max_score[i])

        self.all_species = new_all_species
        self.species_seed_organism = new_seed_organisms
        self.species_last_improvement = species_last_improvement
        self.species_max_score = species_max_score

    def step_generation(self):
        self.get_stats()
        self.__evaluate_population(self.population)
        offspring_per_species = self.__assign_offspring(self.all_species)     # Assign offspring count to each species

        new_organisms: List[Organism] = list()
        new_organisms_species_index: List[int] = list()
        for species_index in range(len(self.all_species)):                    # Crossover and mutate organisms in each species
            if len(self.all_species[species_index]) > 0:
                species_offspring = self.__create_offspring(self.all_species[species_index], offspring_per_species[species_index])
                for organism in species_offspring:
                    new_organisms.append(organism)
                    new_organisms_species_index.append(species_index)

        self.__evaluate_population(new_organisms)                             # Evaluate new organisms
        all_organisms = self.population + new_organisms                       # Join the old population with the new population
        all_scores = np.array([organism.fitness for organism in all_organisms])

        if np.random.ranf() < self.hparams['prob_multiobjective']:
            organism_index_rank = nsga_sort(all_scores)                       # Get positions of organisms in the sorted array
        else:
            organism_index_rank = np.argsort(-all_scores[:, 0])
        for i in range(len(organism_index_rank)):
            all_organisms[i].rank = organism_index_rank[i]

        self.population = [None] * self.hparams['population_size']
        for i in range(len(organism_index_rank)):                            # Sort organisms in O(n)
            if organism_index_rank[i] < self.hparams['population_size']:
                self.population[organism_index_rank[i]] = all_organisms[i]
                self.population[organism_index_rank[i]].assign_species(None)

        self.__speciate_population()                                         # Assign species to newly added organisms
        self.__remove_empty_species()
        self.__speciate_population()                                         # Assign species again

        self.generation_number += 1

