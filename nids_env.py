import gym
from gym import spaces
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score

class NIDSEnv(gym.Env):
    def __init__(self, data, labels):
        super(NIDSEnv, self).__init__()

        self.data = data
        self.labels = labels
        self.current_index = 0

        self.num_features = data.shape[1]

        self.action_space = spaces.Discrete(2) # 0: Benign, 1: Malicious
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_features,), dtype=np.float32)

    def reset(self):
        self.current_index = 0
        return self.data[self.current_index]

    def step(self, action):
        done = False
        true_label = self.labels[self.current_index]
        state = self.data[self.current_index]

        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0
        precision = 0
        recall = 0

        reward = 0
        if action == true_label:
            if action == 1:
                reward = +2  # Correctly detected attack
            else:
                reward = +1  # Correctly detected normal
        else:
            if action == 1:
                reward = -1  # False positive (benign misclassified as attack)
            else:
                reward = -2  # False negative (attack missed)

        self.current_index += 1
        if self.current_index >= len(self.data):
            done = True
            next_state = np.zeros(self.data.shape[1])
        else:
            next_state = self.data[self.current_index]

        return reward, next_state, done

    def render(self, mode='human'):
        pass