import matplotlib.pyplot as plt

from genetic_algorithm import GeneticAlgorithm
from evaluator import Evaluator
import time
import gym
import  optuna

from utility.env_utility import preview
preview('Acrobot-v1')
environment = gym.make('Acrobot-v1')


def objective(trial):
    hparams = {
        'population_size': trial.suggest_int('population_size', 20, 200, 10),
        'gene_coeff': trial.suggest_float('gene_coeff', 0.0, 1.0),
        'weight_coeff': trial.suggest_float('gene_coeff', 0.0, 1.0),
        'species_threshold': trial.suggest_float('gene_coeff', 0.0, 1.0),
        'prob_mutate': trial.suggest_float('gene_coeff', 0.0, 1.0),
        'prob_mutate_add_neuron': 0.25,
        'prob_mutate_add_connection': 0.25,
        'prob_mutate_change_activation': 0.5,
        'dieoff_fraction': trial.suggest_float('gene_coeff', 0.0, 1.0),
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

study = optuna.load_study(
    storage='mysql://89.216.80.18:3306/recurrent-wann1-hparams',
    # pruner=optuna.pruners.HyperbandPruner(
    #     min_resource=1, max_resource=50, reduction_factor=3),
    # sampler=optuna.samplers.TPESampler(),
    study_name='recurrent-wann1'
)
study.optimize(objective, n_trials=20)