import numpy as np
import gym
from recurrent_neural_network import NeuralNetwork

class Evaluator:
    def __init__(self, eval_env, hparams):
        self.eval_env = eval_env
        self.hparams = hparams

    def evaluate_organism(self, organism, render_env = False):
        avg_reward = 0.0
        # all_episode_actions = [list() for _ in range(self.hparams['eval_episodes'])]
        # all_episode_obs = [list() for _ in range(self.hparams['eval_episodes'])]

        for i_episode in range(self.hparams['eval_episodes']):
            observation = self.eval_env.reset()
            total_reward = 0.0
            for t in range(200):
                if render_env:
                    self.eval_env.render()
                shared_weight_value = self.hparams['eval_weights'][i_episode]
                nn: NeuralNetwork = organism.neural_net
                nn.set_shared_weight(shared_weight_value)
                nn.clear_network()
                nn.set_input(observation)
                output_layer = nn.compute_activations()
                action = np.argmax(output_layer)

                # all_episode_actions[i_episode].append(action)
                # all_episode_obs[i_episode].append(observation)

                observation, reward, done, info = self.eval_env.step(action)
                total_reward += reward
                if done:
                    #print("Episode finished after {} timesteps".format(t + 1))
                    break
            avg_reward += total_reward / self.hparams['eval_episodes']
        self.eval_env.close()
        return np.array([avg_reward, min(-(len(organism.gene_ids) - organism.start_gene_count), -1)])

    def get_objective_count(self):
        return 2
