import math

import numpy as np
import gym
import progressbar
from mpi4py import MPI
from typing import List

from organism import Organism
from recurrent_neural_network import NeuralNetwork
from utility.utility_functions import best_int_split

class Evaluator:
    def __init__(self, eval_env, hparams):
        self.eval_env = eval_env
        self.hparams = hparams

    def evaluate_organism(self, organism, render_env=False):

        different_weight_count = len(self.hparams['eval_weights'])
        total_episode_cnt = self.hparams['eval_episodes']*different_weight_count
        avg_reward = np.zeros(different_weight_count)
        for weight_index in range(different_weight_count):
            shared_weight_value = self.hparams['eval_weights'][weight_index]
            nn: NeuralNetwork = organism.neural_net
            nn.set_shared_weight(shared_weight_value)
            nn.clear_network()

            for _ in range(self.hparams['eval_episodes']):
                observation = self.eval_env.reset()
                total_reward = 0.0
                for t in range(1000):
                    if render_env:
                        self.eval_env.render()
                    nn.clear_network()
                    nn.set_input(observation)
                    output_layer = nn.compute_activations()
                    if math.isnan(output_layer):
                        output_layer = np.array([0])
                    action = output_layer

                    observation, reward, done, info = self.eval_env.step(action)
                    total_reward += reward
                    if done:
                        #print("Episode finished after {} timesteps".format(t + 1))
                        break
                avg_reward[weight_index] += total_reward / self.hparams['eval_episodes']
        self.eval_env.close()
        return np.array([avg_reward.mean(), avg_reward.max(), 1.0/max((len(organism.gene_ids),1))])

    def get_objective_count(self):
        return 3

    def evaluate_population_parallel(self, population: List[Organism]):
        print('Evaluating population...', flush=True)

        fraction = 1.0 / (self.hparams['thread_count']-1)

        # Split population into equal parts
        population_split = [list() for _ in range(self.hparams['thread_count']-1)]
        population_counts_split = best_int_split([fraction]*len(population_split),len(population))
        j = 0
        for i in range(len(population_split)):
            population_split[i] = population[j:j+population_counts_split[i]]
            j += population_counts_split[i]
        mpi_comm = MPI.COMM_WORLD
        for i in range(len(population_split)):
            population_part = population_split[i]
            mpi_comm.send(population_part, dest=i+1, tag=1)

        scores = np.zeros((1, self.get_objective_count()))
        for i in range(len(population_split)):
            recv_score = mpi_comm.recv(source=i+1, tag=1)
            scores = np.vstack((scores, recv_score))

        for i in range(1, len(scores)):
            population[i-1].fitness = scores[i]
        print(' ', flush=True)
        return scores

    def evaluate_population_serial(self, population: List[Organism]):
        print('Evaluating population...')
        population_fitnesses = np.zeros((len(population), self.get_objective_count()))
        for i in progressbar.progressbar(range(len(population))):
            population[i].fitness = self.evaluate_organism(population[i])
            population_fitnesses[i] = population[i].fitness
        print(' ')

        return population_fitnesses
