import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.0001, gamma=0.95, epsilon=1.0, epsilon_decay=0.998,
                 epsilon_min=0.02, batch_size=128, memory_size=50000, device=None):
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
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),  # Added dropout to prevent overfitting
            nn.Linear(256, 128),
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

    def remember(self, state, action, reward, next_state, done, label):
        if state.ndim != 1 or next_state.ndim != 1:
            print("❌ Bad shape found!")
            return  # Skip storing this faulty sample

        self.memory.append((state, action, reward, next_state, done, label))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Now using the 6th element (label) for balancing
        benign_samples = [ex for ex in self.memory if ex[5] == 0]
        attack_samples = [ex for ex in self.memory if ex[5] == 1]

        half_batch = self.batch_size // 2

        if len(benign_samples) >= half_batch and len(attack_samples) >= half_batch:
            batch = random.sample(benign_samples, half_batch) + random.sample(attack_samples, half_batch)
        else:
            batch = random.sample(self.memory, self.batch_size)

        random.shuffle(batch)

        states, actions, rewards, next_states, dones, labels = zip(*batch)

        try:
            states = torch.FloatTensor(np.array(states)).to(self.device)
            actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
            next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
            dones = torch.BoolTensor(dones).unsqueeze(1).to(self.device)
        except Exception as e:
            print(f"Failed to convert to batch tensors: {e}")
            return

        curr_q = self.policy_net(states).gather(1, actions)
        
        # Double DQN Logic
        next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
        next_q = self.target_net(next_states).gather(1, next_actions).detach()
        
        target_q = rewards + (1 - dones.float()) * self.gamma * next_q

        loss = self.loss_fn(curr_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # polyak averaging - reduced tau for more stability
        tau = 0.005
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

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