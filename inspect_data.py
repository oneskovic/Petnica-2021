import pickle
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import time
import gym
from organism import Organism
from evaluator import Evaluator
from utility.network_drawing_temp import draw_neural_net

plt.ion()
while True:
    list_of_files = sorted(glob.glob('data/cartpole_swingup/recurrent wann run 2021_09_08 00_39_10/*'), key=os.path.getctime)
    #list_of_files = sorted(glob.glob('C:/Users/Ognjen/Desktop/Petnica-2021-82eade927a359af0f4e5a7cd820673f750fccd8a/data/cartpole_swingup/run 2021_08_21 20_01_22/*'), key=os.path.getctime)
    latest_file_name = list_of_files[-1]
    data_file = open(latest_file_name, 'rb')
    logger = pickle.load(data_file)

    evaluator = logger.logged_values['problem_params'][0]['evaluator']
    last_gen = list(logger.logged_values['best_organism'].keys())[-1]
    evaluator.hparams['eval_episodes'] = 1
    best_organism: Organism = logger.logged_values['best_organism'][last_gen]
    # evaluator.evaluate_organism(best_organism, True)
    # draw_neural_net(best_organism.neural_net)
    # plt.show()
    # plt.pause(0.05)
    # time.sleep(1)
    # plt.cla()
    # continue

    max_scores = list()
    max_max_scores = list()
    avg_scores = list()
    best_organisms = list()
    for generation_number in range(len(logger.logged_values['scores'].keys())):
        generation = logger.logged_values['scores'][generation_number]
        max_score = -10000
        avg_score = 0.0
        max_max_score = -100000
        for i in range(len(generation)):
            max_score = max(max_score, generation[i][0])
            avg_score += generation[i][0] / len(generation)
            max_max_score = max(max_max_score, generation[i][1])
        avg_scores.append(avg_score)
        max_scores.append(max_score)
        max_max_scores.append(max_max_score)
        best_organisms.append(logger.logged_values['best_organism'][generation_number])

    plt.cla()
    plt.plot(max_scores, c='blue', label='Max scores')
    plt.plot(avg_scores, c='red', label='Average scores')
    plt.plot(max_max_scores, c='green', label='Peak scores')
    ax = plt.gca()
    ax.legend()
    plt.show()
    plt.pause(0.05)
    time.sleep(0.3)

    #print(' ')