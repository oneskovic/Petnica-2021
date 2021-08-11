import pickle
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import time
import gym
from evaluator import Evaluator

plt.ion()
while True:
    list_of_files = sorted(glob.glob('data/run 2021_08_10 17_34_12/*'),key=os.path.getctime)
    latest_file_name = list_of_files[-1]
    data_file = open(latest_file_name, 'rb')
    logger = pickle.load(data_file)

    evaluator = logger.logged_values['problem_params'][0]['evaluator']
    last_gen = list(logger.logged_values['best_organism'].keys())[-1]
    #evaluator.evaluate_organism(logger.logged_values['best_organism'][last_gen],True)

    plt.subplot(211)
    plt.cla()
    plt.plot([len(v) for v in logger.logged_values['species_sizes'].values()])
    max_scores = list()
    for generation in logger.logged_values['species_scores'].values():
        max_score = -10000
        for spec in generation:
            for i in range(len(spec)):
                max_score = max(max_score, spec[i][0])
        max_scores.append(max_score)
    plt.subplot(212)
    plt.cla()
    plt.plot(max_scores)
    plt.show()
    plt.pause(0.05)
    time.sleep(0.3)

    #print(' ')