from genetic_algorithm_utility import GeneticAlgorithm
from evaluator import Evaluator

hparams = {
    'population_size': 10,
    'gene_coeff': 0.7,
    'weight_coeff': 0.3,
    'species_threshold': 0.5,
    'prob_mutate': 0.3,
    'dieoff_fraction': 0.3
}
problem_params = {
    'input_layer_size': 4,
    'output_layer_size': 2,
    'evaluator': Evaluator(None)
}
ga = GeneticAlgorithm(hparams,problem_params)
for _ in range(1000):
    ga.step_generation()