import matplotlib.pyplot as plt

from genetic_algorithm import GeneticAlgorithm
from evaluator import Evaluator
import time
import gym

hparams = {
    'population_size': 10,
    'gene_coeff': 0.7,
    'weight_coeff': 0.3,
    'species_threshold': 0.5,
    'prob_mutate': 0.7,
    'dieoff_fraction': 0.3,
    'eval_episodes': 10
}

from preview_environment import preview
preview('Acrobot-v1')
environment = gym.make('Acrobot-v1')
problem_params = {
    'input_layer_size': 6,
    'output_layer_size': 3,
    'evaluator': Evaluator(environment,hparams)
}

generation_cnt = 100
population_size = 100

hparams['population_size'] = population_size
max_fit_list = list()
ga = GeneticAlgorithm(hparams, problem_params)
for generation_number in range(generation_cnt):
    start_time = time.time()

    print(f'Starting generation {generation_number}')
    genome_size = ga.step_generation()
    max_fitness = ga.population[0].fitness[0]
    for organism in ga.population:
        max_fitness = max(max_fitness, organism.fitness[0])
    print('')
    print(f'Max fitness {max_fitness}')
    print(f'Current genome size: {genome_size}')
    print(f'Elapsed time: {round(time.time()-start_time,2)}')
    max_fit_list.append(max_fitness)

plt.plot(max_fit_list)
plt.show()