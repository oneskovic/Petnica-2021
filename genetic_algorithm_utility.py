import numpy as np
from organism import Organism
from copy import deepcopy


class GeneticAlgorithm:
    def __init__(self, hparams, problem_params):
        self.hparams = hparams
        self.all_species = list()
        self.species_seed_organism = list()
        self.problem_params = problem_params
        self.init_population(hparams['population_size'])

    def init_population(self, population_size):
        input_layer_size = self.problem_params['input_layer_size']
        output_layer_size = self.problem_params['output_layer_size']
        self.population = [Organism(input_layer_size, output_layer_size) for _ in range(population_size)]

    def __assign_species(self, organism_index):
        was_assigned_species = False
        for species_index in range(len(self.all_species)):
            dist = self.population[organism_index].get_distance_from(self.species_seed_organism[species_index],
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
        """Divides a total into integer shares that best reflects ratio
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
        return intSplit

    def create_offspring(self, species, offspring_count):
        offspring = list()
        for _ in range(2*offspring_count):
            if np.random.ranf() < self.hparams['prob_mutate']:
                organism_index = np.random.choice(species)
                self.population[organism_index].mutate()
                offspring.append(deepcopy(self.population[organism_index]))
            else:
                parent_indices = np.random.choice(species, 2)
                parent1 = self.population[parent_indices[0]]
                parent2 = self.population[parent_indices[1]]
                offspring.append(parent1.crossover(parent2))
        return offspring

    def assign_offspring(self, all_species):
        avg_scores = np.zeros(len(all_species))
        for species_index in range(len(all_species)):
            total_score = 0.0
            for organism_index in all_species[species_index]:
                total_score += self.problem_params['evaluator'].evaluate_organism(self.population[organism_index])
            avg_scores[species_index] = total_score / len(all_species[species_index])

        offspring_count = self.best_int_split(avg_scores, self.hparams['population_size'])
        return offspring_count

    def __evaluate_population(self):
        population_fitnesses = np.zeros(len(self.population))
        for i in range(len(self.population)):
            self.population[i].assign_fitness(self.problem_params['evaluator'].evaluate_organism(self.population[i]))
            population_fitnesses[i] = self.population[i].fitness
        return population_fitnesses

    def remove_worst_from_population(self):
        fitnesses = self.__evaluate_population()
        remove_cnt = len(self.population) * self.hparams['dieoff_fraction']
        remove_cnt = int(remove_cnt)

        elite_organisms = list()
        sorted_indices = np.argsort(fitnesses)
        for i in range(len(self.population) - remove_cnt):
            elite_organisms.append(self.population[sorted_indices[i]])

        self.all_species = [[] for _ in range(len(self.all_species))]
        self.species_seed_organism = [None] * len(self.species_seed_organism)
        for organism_index in range(len(elite_organisms)):
            species_id = self.population[organism_index].species_id
            if self.population[organism_index].is_seed_organism:
                self.species_seed_organism[species_id] = self.population[organism_index]
            self.all_species[species_id].append(self.population[organism_index])
        for species_id in range(len(self.all_species)):
            if len(self.all_species[species_id]) > 0 and self.species_seed_organism[species_id] is None:
                self.species_seed_organism = self.population[self.all_species[species_id][0]]
        self.population = elite_organisms

    def step_generation(self):
        self.remove_worst_from_population()
        offspring_per_species = self.assign_offspring(self.all_species)
        for species_index in range(len(self.all_species)):
            if self.all_species[species_index]:     # If the species is not empty
                self.all_species[species_index] = self.create_offspring(self.all_species[species_index],offspring_per_species[species_index])
        # TODO: Finish implementation ...


