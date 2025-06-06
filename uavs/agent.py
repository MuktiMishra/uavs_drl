import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # self.net = nn.Sequential(
        #     nn.Linear(input_dim, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, output_dim)
        # )
        self.net = nn.Sequential(
            nn.Linear(5, 128),  
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )


    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, state_dim, action_dim, config):
        self.model = QNetwork(state_dim, action_dim)
        self.target_model = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config["lr"])
        self.criteria = nn.MSELoss()
        self.memory = deque(maxlen=config["memory_size"])
        self.batch_size = config["batch_size"]
        self.gamma = config["gamma"]
        self.epsilon = config["epsilon_start"]
        self.epsilon_decay = config["epsilon_decay"]
        self.epsilon_min = config["epsilon_min"]
        self.update_target_every = config["update_target_every"]
        self.step_count = 0

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, 4)  # 5 actions including 'stay'
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.model(state)
        return torch.argmax(q_vals).item()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones).unsqueeze(1)

        current_q = self.model(states).gather(1, actions)
        next_q = self.target_model(next_states).max(1)[0].unsqueeze(1)
        expected_q = rewards + self.gamma * next_q * (~dones)

        loss = self.criteria(current_q, expected_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
