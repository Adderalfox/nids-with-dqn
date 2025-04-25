import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995,
                 epsilon_min=0.01, batch_size=64, memory_size=10000, device=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.memory = deque(maxlen=memory_size)

        self.policy_net = self._build_network().to(self.device)
        self.target_net = self._build_network().to(self.device)
        self.update_target_network()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr= self.lr)
        self.loss_fn = nn.MSELoss()

    def _build_network(self):
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim)
        )

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        if state.ndim != 1 or next_state.ndim != 1:
            print("❌ Bad shape found!")
            print(f"State shape: {state.shape}, Next state shape: {next_state.shape}")
            return  # Skip storing this faulty sample

        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # states = torch.FloatTensor(np.array(states)).to(self.device)
        # actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        # rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        # next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        # dones = torch.BoolTensor(dones).unsqueeze(1).to(self.device)

        for i, (s, ns) in enumerate(zip(states, next_states)):
            if not isinstance(s, np.ndarray):
                print(f"State at index {i} is not a numpy array: {type(s)}")
            if not isinstance(ns, np.ndarray):
                print(f"Next state at index {i} is not a numpy array: {type(ns)}")
            if s.shape != ns.shape:
                print(f"Mismatch at index {i}: state {s.shape}, next_state {ns.shape}")
            if len(s.shape) != 1:
                print(f"Bad shape at index {i}: state shape {s.shape}")

        try:
            states = torch.FloatTensor(np.array(states)).to(self.device)
            actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
            next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
            dones = torch.BoolTensor(dones).unsqueeze(1).to(self.device)
        except Exception as e:
            print("❌ Failed to convert to batch tensors!")
            print(f"Error: {e}")
            print(f"Sampled batch shapes:")
            for s in states:
                print(np.array(s).shape)
            raise e

        curr_q = self.policy_net(states).gather(1, actions)

        # .max(1, keepdim=True) ---> getting max q value from actions column
        next_q = self.target_net(next_states).max(1, keepdim=True)[0].detach()
        target_q = rewards + (1 - dones.float()) * self.gamma * next_q

        loss = self.loss_fn(curr_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay

    def save_checkpoint(self, name):
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, name)

    def load(self, name):
        checkpoint = torch.load(name)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']