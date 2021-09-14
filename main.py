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
from cartpole_swingup import CartPoleSwingUpEnv

environment = CartPoleSwingUpEnv()
thread_count = 14
num_trials = 300
#preview(environment)


def train(hparams):
    problem_params = {
        'input_layer_size': 5,
        'output_layer_size': 1,
        'evaluator': Evaluator(environment, hparams)
    }
    generation_cnt = 150

    logger = Logger(detalied=False)
    ga = GeneticAlgorithm(hparams, problem_params, logger)

    timestamp = datetime.datetime.now().strftime('%Y_%m_%d %H_%M_%S')
    type_str = "recurrent" if hparams['recurrent_nets'] else "feedforward"
    run_folder = f'data/cartpole_swingup/{type_str} wann run {timestamp}'
    os.makedirs(run_folder)
    for generation_number in range(generation_cnt):
        print(f'Starting generation {generation_number}', flush=True)
        start_time = time.time()
        ga.step_generation()
        file_name = f'{run_folder}/data generation {generation_number}.pk1'
        logger_out_file = open(file_name, 'wb+')
        pickle.dump(logger, logger_out_file, pickle.HIGHEST_PROTOCOL)
        logger_out_file.close()
        end_time = time.time()
        print(f'Time for generation {generation_number}: {round(end_time-start_time,2)} seconds')


def objective(trial):
    x = []
    mutation_names = ['mutate_add_neuron', 'mutate_add_connection', 'mutate_remove_connection', 'mutate_change_activation']
    num_mutations = len(mutation_names)
    for i in range(num_mutations):
        x.append(- np.log(trial.suggest_float(f"x_{mutation_names[i]}", 0, 1)))

    p = []
    for i in range(num_mutations):
        p.append(x[i] / sum(x))

    for i in range(num_mutations):
        trial.set_user_attr(f"p_{mutation_names[i]}", p[i])

    hparams = {
        'population_size': 256,
        'init_connection_fraction': trial.suggest_float('init_connection_fraction', 0.0, 1.0),
        'init_connection_prob': trial.suggest_float('init_connection_prob', 0.0, 1.0),
        'gene_coeff': 1.0,  # Weigh the importance of gene differences
        'weight_coeff': 0.0,  # Weight agnostic, so leave at zero
        'prob_multiobjective': 0.8,
        'prob_mutate': 1.0,  # Probability of mutation
        'prob_mutate_add_neuron': p[0],  # If a mutation occurs the probability of adding a new neuron
        'prob_mutate_add_connection': p[1],  # If a mutation occurs the probability of adding a new connection
        'prob_remove_connection': p[2],
        'prob_mutate_change_activation': p[3],  # If a mutation occurs the probability of changing a neuron activation
        'dieoff_fraction': 0.2,  # The fraction of population that is discarded once sorted
        'elite_fraction': 0.2,  # The fraction of population that is kept unchanged once sorted
        'offspring_weighing': 'linear',  # Options: linear, exponential (1/x)
        'tournament_size': 8,  # The size of tournament used in parent selection
        'eval_episodes': 3,
        'eval_weights': [-2, -1, -0.5, 0.5, 1, 2],
        'thread_count': thread_count,
        'recurrent_nets': True,
        'random_seed': np.random.randint(np.iinfo(int).max)
    }

    problem_params = {
        'input_layer_size': 5,
        'output_layer_size': 1,
        'evaluator': Evaluator(environment, hparams)
    }
    generation_cnt = 150

    logger = Logger(detalied=False)
    ga = GeneticAlgorithm(hparams, problem_params, logger)

    timestamp = datetime.datetime.now().strftime('%Y_%m_%d %H_%M_%S')
    type_str = "recurrent" if hparams['recurrent_nets'] else "feedforward"
    run_folder = f'data/cartpole_swingup/{type_str} wann run {timestamp}'
    os.makedirs(run_folder)

    max_fitness = -1
    for generation_number in range(generation_cnt):
        print(f'Starting generation {generation_number}', flush=True)
        start_time = time.time()

        ga.step_generation()

        file_name = f'{run_folder}/data generation {generation_number}.pk1'
        logger_out_file = open(file_name, 'wb+')
        pickle.dump(logger, logger_out_file, pickle.HIGHEST_PROTOCOL)
        logger_out_file.close()
        end_time = time.time()
        print(f'Time for generation {generation_number}: {round(end_time - start_time, 2)} seconds')

        peak_scores = [org.fitness[1] for org in logger.logged_values['best_organism'].values()]
        max_fitness = max(max_fitness, max(peak_scores))
        trial.report(max_fitness, generation_number)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return max_fitness


mpi_comm = MPI.COMM_WORLD
mpi_size = mpi_comm.Get_size()
mpi_rank = mpi_comm.Get_rank()

if thread_count == 1 or mpi_rank == 0:
    if thread_count > 1:
        print(f'Master at: {os.getpid()}')
    study = optuna.create_study(
        study_name='recurrent-wann-cartpole',
        storage='mysql://89.216.80.18:3306/recurrent-wann1-hparams',
        load_if_exists=True,
        pruner=optuna.pruners.HyperbandPruner(min_resource=20, max_resource=150, reduction_factor=3),
        sampler=optuna.samplers.TPESampler(),
        direction='maximize'
    )
    study.optimize(objective, n_trials=num_trials)
else:
    while True:
        population_to_eval = mpi_comm.recv(source=0, tag=1)
        hparams = mpi_comm.recv(source=0, tag=2)
        evaluator = Evaluator(environment, hparams)
        scores = np.zeros((len(population_to_eval), evaluator.get_objective_count()))
        for i in range(len(population_to_eval)):
            scores[i] = evaluator.evaluate_organism(population_to_eval[i])
        mpi_comm.send(scores, dest=0, tag=1)

# fig = optuna.visualization.plot_parallel_coordinate(study)
# fig.show()


# hparams = {
#     'population_size': 256,
#     'init_connection_fraction': 1.0,
#     'init_connection_prob': 0.5,
#     'gene_coeff': 1.0,                      # Weigh the importance of gene differences
#     'weight_coeff': 0.0,                    # Weight agnostic, so leave at zero
#     'prob_multiobjective': 0.8,
#     'prob_mutate': 1.0,                     # Probability of mutation
#     'prob_mutate_add_neuron': 0.25,         # If a mutation occurs the probability of adding a new neuron
#     'prob_mutate_add_connection': 0.125,    # If a mutation occurs the probability of adding a new connection
#     'prob_remove_connection': 0.125,
#     'prob_mutate_change_activation': 0.5,   # If a mutation occurs the probability of changing a neuron activation
#     'dieoff_fraction': 0.2,                 # The fraction of population that is discarded once sorted
#     'elite_fraction': 0.2,                  # The fraction of population that is kept unchanged once sorted
#     'offspring_weighing': 'linear',         # Options: linear, exponential (1/x)
#     'tournament_size': 8,                   # The size of tournament used in parent selection
#     'eval_episodes': 3,
#     'eval_weights': [-2, -1, -0.5, 0.5, 1, 2],
#     'thread_count': 14,
#     'recurrent_nets': True,
#     'random_seed': np.random.randint(np.iinfo(int).max)
# }
