import matplotlib.pyplot as plt
import numpy as np

from genetic_algorithm import GeneticAlgorithm
from evaluator import Evaluator
import time
import datetime
import pickle
import gym
import optuna
import os
from utility.logger import Logger
from mpi4py import MPI

from utility.env_utility import preview
preview('Pendulum-v0')
environment = gym.make('Pendulum-v0')


def train(hparams):
    problem_params = {
        'input_layer_size': 6,
        'output_layer_size': 3,
        'evaluator': Evaluator(environment, hparams)
    }
    generation_cnt = 150

    logger = Logger()
    ga = GeneticAlgorithm(hparams, problem_params, logger)

    timestamp = datetime.datetime.now().strftime('%Y_%m_%d %H_%M_%S')
    run_folder = f'data/run {timestamp}'
    os.makedirs(run_folder)
    for generation_number in range(generation_cnt):
        print(f'Starting generation {generation_number}', flush=True)
        ga.step_generation()
        file_name = f'{run_folder}/data generation {generation_number}.pk1'
        logger_out_file = open(file_name, 'wb+')
        pickle.dump(logger, logger_out_file, pickle.HIGHEST_PROTOCOL)
        logger_out_file.close()

def objective(trial):
    hparams = {
        'population_size': trial.suggest_int('population_size', 20, 200, 10),
        'gene_coeff': trial.suggest_float('gene_coeff', 0.0, 1.0),
        'weight_coeff': trial.suggest_float('weight_coeff', 0.0, 1.0),
        'species_threshold': trial.suggest_float('species_threshold', 0.0, 1.0),
        'prob_mutate': trial.suggest_float('prob_mutate', 0.0, 1.0),
        'prob_mutate_add_neuron': 0.25,
        'prob_mutate_add_connection': 0.25,
        'prob_mutate_change_activation': 0.5,
        'dieoff_fraction': trial.suggest_float('dieoff_fraction', 0.0, 1.0),
        'eval_episodes': 10,
        'thread_count': 8
    }
    problem_params = {
        'input_layer_size': 6,
        'output_layer_size': 3,
        'evaluator': Evaluator(environment, hparams)
    }


    generation_cnt = 100
    ga = GeneticAlgorithm(hparams, problem_params)
    max_fitness = -1
    for generation_number in range(generation_cnt):

        genome_size = ga.step_generation()
        max_fitness = ga.population[0].fitness[0]
        best_organism = ga.population[0]
        for organism in ga.population:
            if organism.fitness[0] > max_fitness:
                max_fitness = organism.fitness[0]
                best_organism = organism

        print(f'Average genome size: {genome_size}')
        print(f'Max fitness: {max_fitness}')
        # problem_params['evaluator'].evaluate_organism(best_organism, render_env=True)
        trial.report(max_fitness, generation_number)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return max_fitness
#
# study = optuna.load_study(
#     storage='mysql://89.216.80.18:3306/recurrent-wann1-hparams',
#     # pruner=optuna.pruners.HyperbandPruner(
#     #     min_resource=1, max_resource=50, reduction_factor=3),
#     # sampler=optuna.samplers.TPESampler(),
#     study_name='recurrent-wann1'
# )
# fig = optuna.visualization.plot_parallel_coordinate(study)
# fig.show()

# study.optimize(objective, n_trials=20)

mpi_comm = MPI.COMM_WORLD
mpi_size = mpi_comm.Get_size()
mpi_rank = mpi_comm.Get_rank()

hparams = {
    'population_size': 1000,
    'gene_coeff': 1.0,                      # Weigh the importance of gene differences
    'weight_coeff': 0.0,                    # Unused
    'species_threshold_precision': 0.05,    # Precision when searching for species threshold
    'prob_multiobjective': 0.8,
    'prob_mutate': 1.0,                     # Probability of mutation
    'prob_mutate_add_neuron': 0.25,         # If a mutation occurs the probability of adding a new neuron
    'prob_mutate_add_connection': 0.25,     # If a mutation occurs the probability of adding a new connection
    'prob_mutate_change_activation': 0.5,   # If a mutation occurs the probability of changing a neuron activation
    'dieoff_fraction': 0.0,                 # Unused
    'offspring_weighing': 'linear',         # Options: linear, exponential (1/x)
    'tournament_size': 8,                   # The size of tournament used in parent selection
    'species_count': 1,                     # The upper bound on species count
    'eval_episodes': 3,
    'eval_weights': [-2, -1, -0.5, 0.5, 1, 2],
    'thread_count': 14,
    'mutate_amount': 0.05,                  # Unused
    'stagnation_start': 4                   # Number of generations after which a species will be penalized for stagnation
}

# train(hparams)
if mpi_rank == 0:
    train(hparams)
else:
    while True:
        # print(mpi_rank)
        population_to_eval = mpi_comm.recv(source=0, tag=1)
        # print(f'{mpi_rank} Recieved population to eval of size {len(population_to_eval)}')
        evaluator = Evaluator(environment, hparams)
        scores = np.zeros((len(population_to_eval),evaluator.get_objective_count()))
        for i in range(len(population_to_eval)):
            scores[i] = evaluator.evaluate_organism(population_to_eval[i])
        mpi_comm.send(scores, dest=0, tag=1)
