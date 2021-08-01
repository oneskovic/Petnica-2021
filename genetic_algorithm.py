from typing import List, Any, Union

import numpy as np
from organism import Organism
from nsga_sort import nsga_sort
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
        for _ in range(offspring_count):
            if np.random.ranf() < self.hparams['prob_mutate']:
                organism_index = np.random.choice(species)
                new_organism = deepcopy(self.population[organism_index])
                new_organism.mutate()
                offspring.append(new_organism)
            else:
                parent_indices = np.random.choice(species, 2)
                parent1 = self.population[parent_indices[0]]
                parent2 = self.population[parent_indices[1]]
                offspring.append(parent1.crossover(parent2))
        return offspring

    def __speciate_population(self):
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

    def __evaluate_population(self, population):
        print('Evaluating population...')
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
        fronts = nsga_sort(all_scores)                                        # Get fronts from non-domination sort
        front_organism_index_map: List[List[int]] = [list() for _ in range(max(fronts)+1)]
        for i in range(len(fronts)):
            front_organism_index_map[fronts[i]].append(i)

        self.all_species = [[] for _ in range(len(self.all_species))]         # Clear all species
        self.population = list()                                              # Clear the population

        added_cnt = 0
        while added_cnt < self.hparams['population_size']:                     # Add the best organisms back into the new population
            for front in front_organism_index_map:
                to_add = min(self.hparams['population_size'] - added_cnt, len(front))
                for i in range(to_add):
                    organism_to_add: Organism = all_organisms[front[i]]
                    self.population.append(organism_to_add)
                    self.all_species[organism_to_add.species_id].append(len(self.population) - 1)
                    added_cnt += 1

        self.__speciate_population()                                           # Assign species to newly added organisms
        total_genome_len = 0
        for organism in self.population:
            total_genome_len += len(organism.gene_ids)
        return total_genome_len / len(self.population)
