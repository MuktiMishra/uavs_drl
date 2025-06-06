import gym
from gym import spaces
import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# -------- Environment --------
class WindyGridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size
        self.window_size = 512

        self.observation_space = spaces.Dict({
            "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
        })
        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([1, 0]),   # Right
            1: np.array([0, 1]),   # Down
            2: np.array([-1, 0]),  # Left
            3: np.array([0, -1])   # Up
        }

        self.render_mode = render_mode
        self.window = None
        self.clock = None

        self.wind_map = {}

    def _generate_wind_map(self):
        self.wind_map.clear()
        for _ in range(self.size):
            x, y = np.random.randint(0, self.size, size=2)
            wind_dir = self._action_to_direction[np.random.choice(4)]
            wind_strength = np.random.uniform(0.2, 0.8)
            self.wind_map[(x, y)] = {"dir": wind_dir, "strength": wind_strength}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._agent_location = np.array([0, 0])
        self._target_location = np.array([
            self.np_random.integers(self.size),
            self.np_random.integers(self.size)
        ])
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = np.array([
                self.np_random.integers(self.size),
                self.np_random.integers(self.size)
            ])

        self._generate_wind_map()

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs()

    def step(self, action):
        intended_direction = self._action_to_direction[action]
        next_location = self._agent_location + intended_direction

        wind = self.wind_map.get(tuple(self._agent_location))
        if wind:
            wind_dir = wind["dir"]
            strength = wind["strength"]
            drift = wind_dir * np.random.choice([0, 1], p=[1 - strength, strength])
            next_location += drift.astype(int)

        self._agent_location = np.clip(next_location, 0, self.size - 1)
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else -0.01  # small penalty for each step

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, False, {}

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size // self.size

        for (x, y), wind in self.wind_map.items():
            start = (x * pix_square_size + pix_square_size // 2,
                     y * pix_square_size + pix_square_size // 2)
            end = start[0] + int(wind["dir"][0] * 15), start[1] + int(wind["dir"][1] * 15)
            pygame.draw.line(canvas, (0, 128, 0), start, end, 3)

        pygame.draw.rect(
            canvas, (255, 0, 0),
            pygame.Rect(pix_square_size * self._target_location,
                        (pix_square_size, pix_square_size))
        )

        pygame.draw.circle(
            canvas, (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size // 3
        )

        for x in range(self.size + 1):
            pygame.draw.line(canvas, 0, (0, x * pix_square_size), (self.window_size, x * pix_square_size), width=1)
            pygame.draw.line(canvas, 0, (x * pix_square_size, 0), (x * pix_square_size, self.window_size), width=1)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def _get_obs(self):
        return {"agent": self._agent_location.copy(), "target": self._target_location.copy()}

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


# -------- DQN Agent --------
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.model = QNetwork(state_dim, action_dim)
        self.target_model = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criteria = nn.MSELoss()
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        self.update_target_every = 10
        self.step_count = 0

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, 3)
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

        if self.step_count % self.update_target_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        self.step_count += 1

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# -------- Training Loop --------
def obs_to_state(obs):
    return np.concatenate([obs["agent"], obs["target"]]) / 4.0

env = WindyGridWorldEnv(render_mode=None, size=5)
agent = DQNAgent(state_dim=4, action_dim=4)
episodes = 500

for episode in range(episodes):
    obs = env.reset()
    state = obs_to_state(obs)
    done = False
    total_reward = 0

    while not done:
        action = agent.act(state)
        next_obs, reward, done, _, _ = env.step(action)
        next_state = obs_to_state(next_obs)

        agent.remember(state, action, reward, next_state, done)
        agent.learn()

        state = next_state
        total_reward += reward

    print(f"Episode {episode+1}: Total Reward = {total_reward:.2f}")

env.close()
