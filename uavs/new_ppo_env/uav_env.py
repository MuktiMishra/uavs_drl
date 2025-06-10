# === uav_env.py ===

import gym
import numpy as np
from gym import spaces
import random
import pygame

class UAVEnv(gym.Env):
    def __init__(self, config):
        super(UAVEnv, self).__init__()

        self.grid_size = config["env_config"]["grid_size"]
        self.max_steps = config["env_config"]["max_steps"]
        self.channels = config["env_config"]["channels"]

        self.fixed_victims = config["env_config"].get("fixed_victims", [])
        self.fixed_obstacles = config["env_config"].get("fixed_obstacles", [])

        self.battery_max = config["env_config"]["battery_max"]
        self.mission_time_max = config["env_config"]["mission_time_max"]

        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(low=0, high=1,
            shape=(self.channels, self.grid_size, self.grid_size), dtype=np.float32)

        self.reset()

    def reset(self):
        self.steps = 0
        self.done = False

        self.coverage = np.zeros((self.grid_size, self.grid_size))
        self.victims = np.zeros((self.grid_size, self.grid_size))
        self.obstacles = np.zeros((self.grid_size, self.grid_size))
        self.risk = np.random.uniform(0, 1, (self.grid_size, self.grid_size))
        self.battery = np.full((self.grid_size, self.grid_size), -1.0)
        self.time = np.full((self.grid_size, self.grid_size), -1.0)

        for v in self.fixed_victims:
            self.victims[v[0], v[1]] = 1

        for o in self.fixed_obstacles:
            self.obstacles[o[0], o[1]] = 1

        self.agent_pos = [0, 0]

        return self._get_obs()

    def _get_obs(self):
        pos_map = np.zeros((self.grid_size, self.grid_size))
        pos_map[self.agent_pos[0], self.agent_pos[1]] = 1

        self.coverage[self.agent_pos[0], self.agent_pos[1]] = 1
        self.battery[self.agent_pos[0], self.agent_pos[1]] = 1.0
        self.time[self.agent_pos[0], self.agent_pos[1]] = 1.0

        obs = np.stack([
            pos_map,
            self.coverage,
            self.victims,
            self.risk,
            self.obstacles,
            self.battery,
            self.time
        ])
        return obs

    def step(self, action):
        self.steps += 1
        old_pos = self.agent_pos[:]

        if action == 4 and self.agent_pos[0] > 0:
            self.agent_pos[0] -= 1
        elif action == 5 and self.agent_pos[0] < self.grid_size - 1:
            self.agent_pos[0] += 1
        elif action == 6 and self.agent_pos[1] > 0:
            self.agent_pos[1] -= 1
        elif action == 7 and self.agent_pos[1] < self.grid_size - 1:
            self.agent_pos[1] += 1

        if self.obstacles[self.agent_pos[0], self.agent_pos[1]] == 1:
            self.agent_pos = old_pos[:]

        reward = 0.0
        if self.victims[self.agent_pos[0], self.agent_pos[1]] == 1:
            reward += 1.0
            self.victims[self.agent_pos[0], self.agent_pos[1]] = 0

        reward -= self.risk[self.agent_pos[0], self.agent_pos[1]] * 0.1

        if self.steps >= self.max_steps:
            self.done = True

        return self._get_obs(), reward, self.done, {}

    def render(self, mode="human"):
        cell_size = 40
        margin = 2
        window_size = self.grid_size * (cell_size + margin) + margin

        if not hasattr(self, "screen"):
            pygame.init()
            self.screen = pygame.display.set_mode((window_size, window_size))
            pygame.display.set_caption("UAV Environment")
            self.clock = pygame.time.Clock()

        self.screen.fill((30, 30, 30))  # Background

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x = j * (cell_size + margin) + margin
                y = i * (cell_size + margin) + margin

                color = (200, 200, 200)  # default = empty cell
                if [i, j] == self.agent_pos:
                    color = (0, 255, 0)  # agent = green
                elif self.obstacles[i, j] == 1:
                    color = (100, 100, 100)  # obstacle = gray
                elif self.victims[i, j] == 1:
                    color = (255, 0, 0)  # victim = red
                elif self.coverage[i, j] == 1:
                    color = (0, 100, 255)  # visited = blue

                pygame.draw.rect(self.screen, color, (x, y, cell_size, cell_size))

        pygame.display.flip()
        self.clock.tick(40)  # FPS (reduce to make agent faster)