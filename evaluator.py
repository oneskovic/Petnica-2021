import numpy as np
class Evaluator:
    def __init__(self, eval_env):
        self.eval_env = eval_env

    def evaluate_organism(self, organism):
        return np.random.ranf(2)

    def get_objective_count(self):
        return 2
