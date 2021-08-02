from typing import List

import numpy as np
from organism import Organism
from utility.nsga_sort import nsga_sort
from copy import deepcopy
import progressbar


class GeneticAlgorithm:
    all_species: List[List[int]]

    def __init__(self, hparams, problem_params):
        self.hparams = hparams
        self.all_species = list()
        self.species_seed_organism = list()
        self.problem_params = problem_params
        self.__init_population(hparams['population_size'])
        self.evaluator_list = [deepcopy(self.problem_params['evaluator']) for _ in range(self.hparams['thread_count'])]

    def __init_population(self, population_size):
        input_layer_size = self.problem_params['input_layer_size']
        output_layer_size = self.problem_params['output_layer_size']
        self.population = [Organism(input_layer_size, output_layer_size) for _ in range(population_size)]
        self.__speciate_population()

    def __assign_species(self, organism_index):
        if self.population[organism_index].species_id is None:
            was_assigned_species = False
            for species_index in range(len(self.all_species)):
                dist = self.population[organism_index].get_distance_from(self.population[self.species_seed_organism[species_index]],
                                                                         self.hparams['gene_coeff'], self.hparams['weight_coeff'])
                if dist <= self.hparams['species_threshold']:
                    self.all_species[species_index].append(organism_index)
                    self.population[organism_index].assign_species(species_index)
                    was_assigned_species = True
                    break
            if not was_assigned_species:
                self.all_species.append(list([organism_index]))
                self.species_seed_organism.append(organism_index)
                self.population[organism_index].assign_species(len(self.all_species)-1)
                self.population[organism_index].assign_seed()

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
        offspring = list()
        parents = np.zeros(2*offspring_count, dtype=np.int)
        for i in range(2*offspring_count):
            parent1 = np.random.choice(species)
            parent2 = np.random.choice(species)
            if (self.population[parent1].fitness > self.population[parent2].fitness).all():
                parents[i] = parent1
            elif (self.population[parent2].fitness > self.population[parent1].fitness).all():
                parents[i] = parent2
            else:
                parents[i] = np.random.choice([parent1,parent2])
        for i in range(0,2*offspring_count,2):
            if np.random.ranf() < self.hparams['prob_mutate']:
                organism_index = parents[i]
                new_organism = deepcopy(self.population[organism_index])
                new_organism.mutate(self.hparams)
                offspring.append(new_organism)
            else:
                parent_indices = [parents[i], parents[i+1]]
                parent1 = self.population[parent_indices[0]]
                parent2 = self.population[parent_indices[1]]
                offspring.append(parent1.crossover(parent2))
        return offspring

    def __speciate_population(self):
        """
        Clears the current species and assigns species to all organisms in self.population.
        """
        self.all_species = [list() for _ in range(len(self.all_species))]
        for organism_index in range(len(self.population)):
            self.__assign_species(organism_index)

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

    def __remove_worst_from_population(self):
        pass
        # fitnesses = self.__evaluate_population(self.population)
        # remove_cnt = len(self.population) * self.hparams['dieoff_fraction']
        # remove_cnt = int(remove_cnt)
        #
        # elite_organisms = list()
        # sorted_indices = np.argsort(fitnesses)
        # for i in range(len(self.population) - remove_cnt):
        #     elite_organisms.append(self.population[sorted_indices[i]])
        #
        # # Clear all species
        # self.all_species = [[] for _ in range(len(self.all_species))]
        # # self.species_seed_organism = [None] * len(self.species_seed_organism)
        #
        # # Fill species with elite organisms
        # for organism_index in range(len(elite_organisms)):
        #     species_id = self.population[organism_index].species_id
        #     if self.population[organism_index].is_seed_organism:
        #         self.species_seed_organism[species_id] = organism_index
        #     self.all_species[species_id].append(organism_index)
        #
        # # for species_id in range(len(self.all_species)):
        # #     if len(self.all_species[species_id]) > 0 and self.species_seed_organism[species_id] is None:
        # #         self.species_seed_organism = self.population[self.all_species[species_id][0]]
        # self.population = elite_organisms

    def step_generation(self):
        self.__evaluate_population(self.population)
        self.__remove_worst_from_population()                                 # Remove the least fit organisms
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
        organism_index_rank = nsga_sort(all_scores)                           # Get positions of organisms in the sorted array
        self.population = [None] * self.hparams['population_size']
        for i in range(len(organism_index_rank)):
            if organism_index_rank[i] < self.hparams['population_size']:
                self.population[organism_index_rank[i]] = all_organisms[i]
                self.population[organism_index_rank[i]].assign_species(None)

        self.__speciate_population()                                           # Assign species to newly added organisms
        total_genome_len = 0
        for organism in self.population:
            total_genome_len += len(organism.gene_ids)
        return total_genome_len / len(self.population)

