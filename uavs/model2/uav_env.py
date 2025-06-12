import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import os
import gym
from gym import spaces

# Set device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

class SimpleUAVEnv(gym.Env):
    def __init__(self):
        super(SimpleUAVEnv, self).__init__()
        self.grid_size = (15, 15)
        self.grid_height, self.grid_width = self.grid_size

        self.channels = 7
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.channels, *self.grid_size), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.max_steps = 50

        self.fixed_victims = [
            (0, 2), (1, 4), (2, 8), (3, 1), (4, 12), (5, 3), (6, 6), (7, 10),
            (8, 14), (9, 7), (10, 5), (11, 2), (12, 13), (13, 0), (14, 4),
            (2, 13), (6, 1), (10, 10), (11, 14), (7, 2), (0, 12), (3, 7)
        ]
        
        self.fixed_obstacles = [
            (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 7), (7, 8),
            (8, 9), (9, 10), (10, 11), (11, 12), (12, 1), (13, 2), (14, 3),
            (1, 13), (4, 8), (9, 0), (13, 10), (5, 14), (6, 0), (7, 5)
        ]

    def reset(self):
        self.step_count = 0
        self.energy_used = 0.0
        self.mission_time = 0.0
        self.risk_score = 0.0
    
        self.coverage_map = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        self.victim_map = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        self.obstacle_map = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)

        self.uav_pos = (0, 0)
        while self.uav_pos in self.fixed_victims or self.uav_pos in self.fixed_obstacles:
            self.uav_pos = (
                np.random.randint(self.grid_height),
                np.random.randint(self.grid_width)
            )

        for (y, x) in self.fixed_victims:
            self.victim_map[y, x] = 1
        for (y, x) in self.fixed_obstacles:
            self.obstacle_map[y, x] = 1

        self.battery = 1.0
        self.time_left = 1.0
        self.risk_map = np.random.rand(self.grid_height, self.grid_width)

        return self.get_state()

    def get_state(self):
        uav_layer = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        uav_layer[self.uav_pos] = 1.0

        battery_layer = np.full((self.grid_height, self.grid_width), self.battery, dtype=np.float32)
        time_layer = np.full((self.grid_height, self.grid_width), self.time_left, dtype=np.float32)
        risk_layer = self.risk_map.copy()

        stacked = np.stack([
            self.coverage_map,
            self.victim_map,
            self.obstacle_map,
            uav_layer,
            battery_layer,
            time_layer,
            risk_layer
        ], axis=0)

        return stacked

    # def step(self, action):
    #     self.step_count += 1

    #     y, x = self.uav_pos
    #     if action == 0 and y > 0: y -= 1
    #     elif action == 1 and y < self.grid_height - 1: y += 1
    #     elif action == 2 and x > 0: x -= 1
    #     elif action == 3 and x < self.grid_width - 1: x += 1
    #     if (y, x) not in self.fixed_obstacles:
    #         self.uav_pos = (y, x)

    #     self.coverage_map[self.uav_pos] = 1.0

    #     E_base = np.random.uniform(0.002, 0.01)
    #     E_action = np.random.uniform(0.001, 0.015)
    #     E_task = np.random.uniform(0.001, 0.02)
    #     E_env = np.random.normal(0.0012, 0.0001)
    #     energy = E_base + E_action + E_task + E_env
    #     self.energy_used += energy

    #     time_taken = np.random.uniform(0.1, 1.0)
    #     self.mission_time += time_taken
    #     self.time_left -= time_taken / self.max_steps

    #     risk = self.risk_map[self.uav_pos]
    #     self.risk_score += risk

    #     done = self.step_count >= self.max_steps or self.battery <= 0 or self.time_left <= 0

    #     E_max, tau_max, R_max = 1.0, self.max_steps, self.max_steps
    #     f2 = 1 - (self.energy_used / E_max)
    #     f3 = 1 - (self.mission_time / tau_max)
    #     f4 = 1 - (self.risk_score / R_max)
    #     reward = (f2 + f3 + f4) / 3

    #     self.battery -= energy
    #     next_state = self.get_state()

    #     return next_state, reward, done, {
    #         'energy_used': self.energy_used,
    #         'mission_time': self.mission_time,
    #         'risk_score': self.risk_score,
    #         'battery': self.battery,
    #         'time_left': self.time_left,
    #         'step': self.step_count,
    #         'f2_energy': f2,
    #         'f3_time': f3,
    #         'f4_risk': f4,
    #         'reward': reward
    #     }

    def step(self, action):
        self.step_count += 1
        prev_pos = self.uav_pos  # Track previous position
    
        y, x = self.uav_pos
        if action == 0 and y > 0: y -= 1
        elif action == 1 and y < self.grid_height - 1: y += 1
        elif action == 2 and x > 0: x -= 1
        elif action == 3 and x < self.grid_width - 1: x += 1
        if (y, x) not in self.fixed_obstacles:
            self.uav_pos = (y, x)
    
        self.coverage_map[self.uav_pos] = 1.0
    
        # Energy, time, and risk logic remains same
        E_base = np.random.uniform(0.002, 0.01)
        E_action = np.random.uniform(0.001, 0.015)
        E_task = np.random.uniform(0.001, 0.02)
        E_env = np.random.normal(0.0012, 0.0001)
        energy = E_base + E_action + E_task + E_env
        self.energy_used += energy
    
        time_taken = np.random.uniform(0.1, 1.0)
        self.mission_time += time_taken
        self.time_left -= time_taken / self.max_steps
    
        risk = self.risk_map[self.uav_pos]
        self.risk_score += risk
    
        done = self.step_count >= self.max_steps or self.battery <= 0 or self.time_left <= 0
    
        # Compute raw reward
        E_max, tau_max, R_max = 1.0, self.max_steps, self.max_steps
        f2 = 1 - (self.energy_used / E_max)
        f3 = 1 - (self.mission_time / tau_max)
        f4 = 1 - (self.risk_score / R_max)
        reward = (f2 + f3 + f4) / 3
    
        # 👇 Add stagnation penalty here
        if self.uav_pos == prev_pos:
            reward -= 0.05  # penalize not moving
    
        self.battery -= energy
        next_state = self.get_state()
    
        return next_state, reward, done, {
            'energy_used': self.energy_used,
            'mission_time': self.mission_time,
            'risk_score': self.risk_score,
            'battery': self.battery,
            'time_left': self.time_left,
            'step': self.step_count,
            'f2_energy': f2,
            'f3_time': f3,
            'f4_risk': f4,
            'reward': reward
        }


class CNNActorCritic(nn.Module):
    def __init__(self, input_channels=7, num_actions=4):
        super(CNNActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8 * 15 * 15, 128)
        self.policy_logits = nn.Linear(128, num_actions)
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.flatten(x)
        x = torch.relu(self.fc(x))
        return self.policy_logits(x), self.value(x)

class PPOAgent:
    def __init__(self, model, optimizer, gamma=0.99):
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma

    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        logits, value = self.model(state_tensor)
        probs = torch.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value

    def update(self, rewards, log_probs, values):
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        log_probs = torch.stack(log_probs).to(device)
        values = torch.cat(values).squeeze().to(device)
        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# === Main Training Loop ===
def train():
    env = SimpleUAVEnv()
    model = CNNActorCritic().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    agent = PPOAgent(model, optimizer)

    num_episodes = 10000
    max_steps = 50
    records = []

    for episode in range(num_episodes):
        state = env.reset()
        log_probs, values, rewards = [], [], []
        total_reward = 0

        for step in range(max_steps):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            records.append({
                'episode': episode,
                'step': step,
                'action': action,
                'reward': reward,
                'energy_used': info['energy_used'],
                'mission_time': info['mission_time'],
                'risk_score': info['risk_score']
            })

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            total_reward += reward
            state = next_state

            if done:
                break

        agent.update(rewards, log_probs, values)

        print(f"Episode {episode + 1}/{num_episodes} - Total Reward: {total_reward:.4f}")


    os.makedirs("model_output", exist_ok=True)
    torch.save(model.state_dict(), "model_output/uav_ppo_cnn_model.pth")
    pd.DataFrame(records).to_csv("model_output/training_log.csv", index=False)

if __name__ == "__main__":
    train()
