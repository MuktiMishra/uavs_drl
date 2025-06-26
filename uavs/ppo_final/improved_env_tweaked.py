# improved_env2.py
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# Constants
GRID_SIZE = 15
UAVS = 2
VICTIM_PERCENTAGE = 0.10
OBSTACLE_PERCENTAGE = 0.10
EPISODES = 3000
MAX_STEPS = 100

# Battery Constants
EBASE = 0.0001
EACTION = 0.00005
ETASK = 0.0001
K_WIND = 0.00001
RISK_WEIGHT = 0.01

# Paths
MODEL_PATH = "ppo_model_tweaked2.pt"
LOG_FILE = "training_logs_tweaked2.xlsx"

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PPO Hyperparameters
GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_EPOCHS = 4
CLIP_EPS = 0.2
ACTOR_LR = 1e-4
CRITIC_LR = 1e-3

# Pygame Setup
pygame.init()
CELL_SIZE = 40
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
clock = pygame.time.Clock()

# Color Constants
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GRAY = (180, 180, 180)
WIND_COLOR = (0, 255, 255)

# Directions
direction_map = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2}

class CNNActorCritic(nn.Module):
    def __init__(self, input_channels=7, num_actions=4):
        super(CNNActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        flattened_size = 8 * GRID_SIZE * GRID_SIZE
        self.actor = nn.Linear(flattened_size, num_actions)
        self.critic = nn.Linear(flattened_size, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)

class GridEnv:
    def __init__(self):
        self.grid_size = GRID_SIZE
        self.reset()

    def reset(self):
        self.coverage = np.zeros((GRID_SIZE, GRID_SIZE))
        self.victim_map = np.zeros((GRID_SIZE, GRID_SIZE))
        self.obstacle_map = np.zeros((GRID_SIZE, GRID_SIZE))
        self.battery_map = np.ones((GRID_SIZE, GRID_SIZE))
        self.time_map = np.ones((GRID_SIZE, GRID_SIZE))
        self.risk_map = np.random.rand(GRID_SIZE, GRID_SIZE)
        self.visit_count = np.zeros((GRID_SIZE, GRID_SIZE))

        self.wind_speed = np.zeros((GRID_SIZE, GRID_SIZE))
        self.wind_direction = np.full((GRID_SIZE, GRID_SIZE), -1)
        wind_cells = int(GRID_SIZE * GRID_SIZE * 0.08)
        all_positions = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)]
        random.shuffle(all_positions)

        for _ in range(wind_cells):
            x, y = all_positions.pop()
            self.wind_speed[x, y] = np.random.rand()
            self.wind_direction[x, y] = np.random.choice([0, 1, 2, 3])
            self.risk_map[x, y] += 0.2 * self.wind_speed[x, y]

        self.total_victims = int(GRID_SIZE * GRID_SIZE * VICTIM_PERCENTAGE)
        for _ in range(self.total_victims):
            x, y = all_positions.pop()
            self.victim_map[x, y] = 1

        for _ in range(int(GRID_SIZE * GRID_SIZE * OBSTACLE_PERCENTAGE)):
            x, y = all_positions.pop()
            self.obstacle_map[x, y] = 1

        self.uav_positions = self.get_unique_positions(UAVS)
        self.batteries = [1.0 for _ in range(UAVS)]
        self.steps_taken = [0 for _ in range(UAVS)]
        self.risks_taken = [0 for _ in range(UAVS)]
        self.done = False
        self.total_time = 0
        return self.get_state()

    def get_unique_positions(self, count):
        positions = set()
        while len(positions) < count:
            pos = (np.random.randint(0, GRID_SIZE), np.random.randint(0, GRID_SIZE))
            if self.obstacle_map[pos] == 0:
                positions.add(pos)
        return list(positions)

    def step(self, actions):
        rewards = []
        new_positions = []
        occupied = set()
        self.total_time += 1

        # First pass: determine all new positions
        for i, (pos, action) in enumerate(zip(self.uav_positions, actions)):
            x, y = pos
            wind_dir = self.wind_direction[x, y]

            if wind_dir != -1:
                if action == wind_dir:
                    pass
                elif action == OPPOSITE[wind_dir]:
                    action = wind_dir
                else:
                    if np.random.rand() < 0.5:
                        action = wind_dir

            new_pos = self._move(pos, action)
            x_new, y_new = new_pos

            wind_cost = self.wind_speed[x_new, y_new] * K_WIND
            delta_E = EBASE + EACTION + ETASK + np.random.normal(0, 0.001) + wind_cost
            self.batteries[i] -= delta_E
            self.steps_taken[i] += 1
            self.risks_taken[i] += self.risk_map[x_new, y_new]
            self.time_map[x_new, y_new] = self.total_time / MAX_STEPS
            self.visit_count[x_new, y_new] += 1

            if self.batteries[i] <= 0 or new_pos in occupied or self.obstacle_map[new_pos] == 1:
                new_positions.append(pos)
            else:
                new_positions.append(new_pos)
                occupied.add(new_pos)
                self.coverage[x_new, y_new] = 1

        # Second pass: reward calculation
        for i, new_pos in enumerate(new_positions):
            x, y = new_pos
            reward = 0.0

            if self.victim_map[x, y] == 1:
                self.victim_map[x, y] = 0
                reward += 10

            if self.coverage[x, y] == 0:
                reward += 1.0
            else:
                reward -= 0.1

            exploration_bonus = (1 - (self.total_time / MAX_STEPS)) * 0.5
            reward += exploration_bonus

            for j, other_pos in enumerate(new_positions):
                if i != j and abs(new_pos[0] - other_pos[0]) <= 1 and abs(new_pos[1] - other_pos[1]) <= 1:
                    reward -= 0.3

            risk_penalty = -RISK_WEIGHT * self.risk_map[x, y]
            energy_efficiency = 1 - (self.batteries[i])
            reward += -energy_efficiency + risk_penalty

            rewards.append(reward)

        self.uav_positions = new_positions
        self.done = np.sum(self.victim_map) == 0 or all(step >= MAX_STEPS for step in self.steps_taken)
        return self.get_state(), rewards, self.done

    def _move(self, pos, action):
        x, y = pos
        if action == 0 and y > 0: y -= 1
        elif action == 1 and y < GRID_SIZE - 1: y += 1
        elif action == 2 and x > 0: x -= 1
        elif action == 3 and x < GRID_SIZE - 1: x += 1
        return (x, y)

    def get_state(self):
        state = np.stack([
            self.coverage,
            self.victim_map,
            self.risk_map,
            self.obstacle_map,
            self._uav_position_channel(),
            self.battery_map,
            self.time_map
        ], axis=0)
        return torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

    def _uav_position_channel(self):
        pos_map = np.zeros((GRID_SIZE, GRID_SIZE))
        for (x, y) in self.uav_positions:
            pos_map[x, y] = 1
        return pos_map

    def render(self):
        screen.fill(WHITE)
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                rect = pygame.Rect(y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if self.obstacle_map[x, y]:
                    pygame.draw.rect(screen, BLACK, rect)
                elif self.victim_map[x, y]:
                    pygame.draw.rect(screen, RED, rect)
                elif self.coverage[x, y]:
                    pygame.draw.rect(screen, GRAY, rect)
                pygame.draw.rect(screen, BLACK, rect, 1)

                dir = self.wind_direction[x, y]
                if dir == 0:
                    pygame.draw.line(screen, WIND_COLOR, (y * CELL_SIZE + 20, x * CELL_SIZE + 10), (y * CELL_SIZE + 20, x * CELL_SIZE), 2)
                elif dir == 1:
                    pygame.draw.line(screen, WIND_COLOR, (y * CELL_SIZE + 20, x * CELL_SIZE + 30), (y * CELL_SIZE + 20, x * CELL_SIZE + 40), 2)
                elif dir == 2:
                    pygame.draw.line(screen, WIND_COLOR, (y * CELL_SIZE + 10, x * CELL_SIZE + 20), (y * CELL_SIZE, x * CELL_SIZE + 20), 2)
                elif dir == 3:
                    pygame.draw.line(screen, WIND_COLOR, (y * CELL_SIZE + 30, x * CELL_SIZE + 20), (y * CELL_SIZE + 40, x * CELL_SIZE + 20), 2)

        for (x, y) in self.uav_positions:
            pygame.draw.circle(screen, BLUE, (y * CELL_SIZE + CELL_SIZE // 2, x * CELL_SIZE + CELL_SIZE // 2), 10)
        pygame.display.flip()
        pygame.time.delay(100)


# (rest of code remains unchanged)




# PPO Agent
class PPOAgent:
    def __init__(self):
        self.model = CNNActorCritic().to(device)
        self.optimizer_actor = optim.Adam(self.model.actor.parameters(), lr=ACTOR_LR)
        self.optimizer_critic = optim.Adam(self.model.critic.parameters(), lr=CRITIC_LR)
        self.memory = []
        if os.path.exists(MODEL_PATH):
            self.model.load_state_dict(torch.load(MODEL_PATH))

    def get_action(self, state):
        logits, _ = self.model(state)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def remember(self, transition):
        self.memory.append(transition)

    def train(self):
        states, actions, rewards, next_states, log_probs, dones = zip(*self.memory)
        self.memory = []

        states = torch.cat(states)
        actions = torch.tensor(actions, device=device)
        rewards = torch.tensor(rewards, device=device, dtype=torch.float32)
        old_log_probs = torch.stack(log_probs).detach()

        _, values = self.model(states)
        values = values.squeeze()

        returns = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + GAMMA * (values[i + 1] if i + 1 < len(values) else 0) - values[i]
            gae = delta + GAMMA * GAE_LAMBDA * gae
            returns.insert(0, gae + values[i])
        returns = torch.tensor(returns, device=device)

        advantages = returns - values.detach()

        actor_loss = []
        critic_loss = []

        for _ in range(PPO_EPOCHS):
            logits, values_pred = self.model(states)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(actions)

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages
            actor_loss_epoch = -torch.min(surr1, surr2).mean() - 0.01 * entropy

            critic_loss_epoch = (returns - values_pred.squeeze()).pow(2).mean()

            self.optimizer_actor.zero_grad()
            actor_loss_epoch.backward(retain_graph=True)
            self.optimizer_actor.step()

            self.optimizer_critic.zero_grad()
            critic_loss_epoch.backward()
            self.optimizer_critic.step()

            actor_loss.append(actor_loss_epoch.item())
            critic_loss.append(critic_loss_epoch.item())

        torch.save(self.model.state_dict(), MODEL_PATH)
        return np.mean(actor_loss), np.mean(critic_loss)

# Main Training Loop
agent = PPOAgent()
env = GridEnv()

actor_losses, critic_losses, coverages, victims_rescued = [], [], [], []

if os.path.exists(LOG_FILE):
    df = pd.read_excel(LOG_FILE)
    actor_losses = df['actor_loss'].tolist()
    critic_losses = df['critic_loss'].tolist()
    coverages = df['coverage'].tolist()
    victims_rescued = df['victims_rescued'].tolist()
else:
    for ep in tqdm(range(EPISODES)):
        state = env.reset()
        done = False
        ep_rewards = []
        steps = 0
        while not done and steps < MAX_STEPS:
            actions = []
            for _ in range(UAVS):
                action, log_prob = agent.get_action(state)
                actions.append(action)
                agent.remember((state, action, 0.1, state, log_prob, done))
            state, rewards, done = env.step(actions)
            ep_rewards.extend(rewards)
            env.render()
            steps += 1

        actor_loss, critic_loss = agent.train()

        coverage_val = np.sum(env.coverage)
        victims_val = env.total_victims - np.sum(env.victim_map)
        print(f"Episode {ep+1}/{EPISODES} | Actor Loss: {actor_loss:.4f} | Critic Loss: {critic_loss:.4f} | Coverage: {coverage_val} | Victims Rescued: {victims_val}")

        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)
        coverages.append(coverage_val)
        victims_rescued.append(victims_val)

    df = pd.DataFrame({
        'actor_loss': actor_losses,
        'critic_loss': critic_losses,
        'coverage': coverages,
        'victims_rescued': victims_rescued
    })
    df.to_excel(LOG_FILE, index=False)

# Plotting
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(actor_losses)
plt.title("Actor Loss")

plt.subplot(2, 2, 2)
plt.plot(critic_losses)
plt.title("Critic Loss")

plt.subplot(2, 2, 3)
plt.plot(coverages)
plt.title("Coverage per Episode")

plt.subplot(2, 2, 4)
plt.plot(victims_rescued)
plt.title("Victims Rescued per Episode")

plt.tight_layout()
plt.show()
pygame.quit()
