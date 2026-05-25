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
        # Start at a random index to ensure diversity across episodes
        self.current_index = np.random.randint(0, len(self.data) - 1)
        return self.data[self.current_index]

    def step(self, action):
        done = False
        true_label = self.labels[self.current_index]
        state = self.data[self.current_index]

        reward = 0
        if action == true_label:
            if action == 1:
                reward = 1.0  # Correctly detected attack
            else:
                reward = 0.5  # Correctly detected normal (less reward than detecting attack)
        else:
            if action == 1:
                reward = -1.0  # False positive
            else:
                reward = -2.0  # False negative (dangerous, higher penalty)

        self.current_index += 1
        if self.current_index >= len(self.data):
            done = True
            next_state = np.zeros(self.data.shape[1])
        else:
            next_state = self.data[self.current_index]

        # Return true_label as part of info or a dedicated field to help with balanced sampling in replay
        return reward, next_state, done, true_label

    def render(self, mode='human'):
        pass