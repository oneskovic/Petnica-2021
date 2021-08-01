import numpy as np
import gym
from recurrent_neural_network import NeuralNetwork

class Evaluator:
    def __init__(self, eval_env, hparams):
        self.eval_env = eval_env
        self.hparams = hparams

    def evaluate_organism(self, organism):
        avg_reward = 0.0
        for i_episode in range(self.hparams['eval_episodes']):
            observation = self.eval_env.reset()
            total_reward = 0.0
            for t in range(200):
                nn: NeuralNetwork = organism.neural_net
                nn.set_input(observation)
                output_layer = nn.compute_activations()
                action = np.argmax(output_layer)
                observation, reward, done, info = self.eval_env.step(action)
                total_reward += reward
                if done:
                    #print("Episode finished after {} timesteps".format(t + 1))
                    break
            avg_reward += total_reward / self.hparams['eval_episodes']
        return np.array([avg_reward,len(organism.gene_ids - organism.start_gene_count)])

    def get_objective_count(self):
        return 2
