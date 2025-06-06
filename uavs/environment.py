# import gym
# from gym import spaces
# import numpy as np
# import pygame
# import yaml
# import random

# class MultiUAVEnv(gym.Env):
#     metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}

#     def __init__(self, config_path="config.yaml", render_mode=None):
#         # Load config
#         with open(config_path, "r") as f:
#             config = yaml.safe_load(f)

#         env_cfg = config["environment"]
#         self.size = env_cfg["grid_size"]
#         self.num_agents = env_cfg["num_agents"]
#         self.num_targets = env_cfg["num_targets"]
#         self.wind_prob = env_cfg["wind_probability"]
#         self.wind_min = env_cfg["wind_strength_min"]
#         self.wind_max = env_cfg["wind_strength_max"]

#         self.render_mode = render_mode
#         self.window_size = 600
#         self.cell_size = self.window_size // self.size

#         self.action_space = spaces.MultiDiscrete([5] * self.num_agents)  # 0:right,1:down,2:left,3:up,4:stay

#         self.observation_space = spaces.Tuple([
#             spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32) for _ in range(self.num_agents)
#         ])

#         self.window = None
#         self.clock = None

#         self._action_to_dir = {
#             0: np.array([1, 0]),   # Right
#             1: np.array([0, 1]),   # Down
#             2: np.array([-1, 0]),  # Left
#             3: np.array([0, -1]),  # Up
#             4: np.array([0, 0])    # Stay
#         }

#         # Track visited cells by all agents (for coverage calculation)
#         self.visited_cells = set()
#         self.steps = 0
#         self.done = False

#         self.reset()

#     def reset(self, seed=None, options=None):
#         self.np_random = np.random.RandomState(seed)

#         # Place agents at random distinct positions
#         self.agent_positions = []
#         taken = set()
#         for _ in range(self.num_agents):
#             while True:
#                 pos = tuple(self.np_random.randint(0, self.size, 2))
#                 if pos not in taken:
#                     taken.add(pos)
#                     self.agent_positions.append(np.array(pos))
#                     break

#         # Place targets randomly, distinct from agents
#         self.targets = set()
#         while len(self.targets) < self.num_targets:
#             pos = tuple(self.np_random.randint(0, self.size, 2))
#             if pos not in taken:
#                 self.targets.add(pos)

#         # Generate wind map: exactly 10% of the cells have wind
#         self.wind_map = {}
#         total_cells = self.size * self.size
#         num_wind_cells = int(total_cells * self.wind_prob)  # This is already 10% from your config
#         all_positions = [(x, y) for x in range(self.size) for y in range(self.size)]
#         wind_positions = self.np_random.choice(len(all_positions), size=num_wind_cells, replace=False)
#         for idx in wind_positions:
#             x, y = all_positions[idx]
#             wind_dir = self._action_to_dir[self.np_random.randint(0, 4)]
#             wind_strength = self.np_random.uniform(self.wind_min, self.wind_max)
#             self.wind_map[(x, y)] = {"dir": wind_dir, "strength": wind_strength}

#         self.visited_cells = set()
#         for pos in self.agent_positions:
#             self.visited_cells.add(tuple(pos))

#         self.steps = 0
#         self.done = False

#         if self.render_mode == "human":
#             self._render_frame()

#         return self._get_obs()

#     def _get_obs(self):
#         observations = []
#         for pos in self.agent_positions:
#             if self.targets:
#                 targets_np = np.array(list(self.targets))
#                 dists = np.linalg.norm(targets_np - pos, axis=1)
#                 nearest_idx = np.argmin(dists)
#                 nearest_target = targets_np[nearest_idx]
#             else:
#                 nearest_target = pos

#             obs = np.concatenate([pos / (self.size - 1), nearest_target / (self.size - 1)]).astype(np.float32)
#             observations.append(obs)
#         return observations

#     def step(self, actions):
#         assert len(actions) == self.num_agents
#         rewards = [0.0] * self.num_agents

#         new_positions = []
#         occupied = set()
#         for i, (pos, action) in enumerate(zip(self.agent_positions, actions)):
#             intended_dir = self._action_to_dir[action]
#             next_pos = pos + intended_dir

#             wind = self.wind_map.get(tuple(pos))
#             if wind:
#                 strength = wind["strength"]
#                 if self.np_random.rand() < strength:
#                     next_pos = next_pos + wind["dir"]

#             next_pos = np.clip(next_pos, 0, self.size - 1)

#             # Prevent collision or overlapping positions
#             if tuple(next_pos) in occupied or tuple(next_pos) in [tuple(p) for p in self.agent_positions]:
#                 next_pos = pos

#             occupied.add(tuple(next_pos))
#             new_positions.append(next_pos)

#         self.agent_positions = new_positions

#         for i, pos in enumerate(self.agent_positions):
#             pos_tuple = tuple(pos)
#             if pos_tuple in self.targets:
#                 rewards[i] += 1.0
#                 self.targets.remove(pos_tuple)

#         # Update visited cells with new positions
#         for pos in self.agent_positions:
#             self.visited_cells.add(tuple(pos))

#         self.steps += 1
#         if len(self.targets) == 0 or self.steps >= 200:
#             self.done = True

#         # Print coverage stats every step (you can comment this if too verbose)
#         coverage = len(self.visited_cells) / (self.size * self.size)
#         print(f"Step: {self.steps} | Visited cells: {len(self.visited_cells)} / {self.size * self.size} "
#               f"({coverage:.2%} coverage)")

#         if self.render_mode == "human":
#             self._render_frame()

#         return self._get_obs(), rewards, self.done, False, {}

#     def render(self):
#         if self.render_mode == "rgb_array":
#             return self._render_frame()
#         elif self.render_mode == "human":
#             self._render_frame()

#     def _render_frame(self):
#         if self.window is None and self.render_mode == "human":
#             pygame.init()
#             pygame.display.init()
#             self.window = pygame.display.set_mode((self.window_size, self.window_size))
#             pygame.display.set_caption("Multi-UAV Windy Grid World")

#         if self.clock is None and self.render_mode == "human":
#             self.clock = pygame.time.Clock()

#         canvas = pygame.Surface((self.window_size, self.window_size))
#         canvas.fill((255, 255, 255))

#         for x in range(self.size + 1):
#             pygame.draw.line(canvas, (200, 200, 200), (x * self.cell_size, 0), (x * self.cell_size, self.window_size))
#             pygame.draw.line(canvas, (200, 200, 200), (0, x * self.cell_size), (self.window_size, x * self.cell_size))

#         for (x, y), wind in self.wind_map.items():
#             start = (x * self.cell_size + self.cell_size // 2, y * self.cell_size + self.cell_size // 2)
#             end = (start[0] + int(wind["dir"][0] * 15), start[1] + int(wind["dir"][1] * 15))
#             pygame.draw.line(canvas, (0, 128, 0), start, end, 3)
#             pygame.draw.circle(canvas, (0, 128, 0), end, 4)

#         for (x, y) in self.targets:
#             rect = pygame.Rect(x * self.cell_size + 5, y * self.cell_size + 5, self.cell_size - 10, self.cell_size - 10)
#             pygame.draw.rect(canvas, (255, 0, 0), rect)

#         colors = [(0, 0, 255), (255, 165, 0), (0, 255, 255), (255, 0, 255)]
#         for i, pos in enumerate(self.agent_positions):
#             center = (int(pos[0] * self.cell_size + self.cell_size / 2), int(pos[1] * self.cell_size + self.cell_size / 2))
#             pygame.draw.circle(canvas, colors[i % len(colors)], center, self.cell_size // 3)

#         if self.render_mode == "human":
#             self.window.blit(canvas, canvas.get_rect())
#             pygame.event.pump()
#             pygame.display.update()
#             self.clock.tick(self.metadata["render_fps"])
#         else:
#             return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

#     def close(self):
#         if self.window:
#             pygame.display.quit()
#             pygame.quit()
import gym
from gym import spaces
import numpy as np
import pygame
import yaml
import random

class MultiUAVEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}

    def __init__(self, config_path="config.yaml", render_mode=None):
        # Load config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        env_cfg = config["environment"]
        self.size = env_cfg["grid_size"]
        self.num_agents = env_cfg["num_agents"]
        self.num_targets = env_cfg["num_targets"]
        self.wind_prob = env_cfg["wind_probability"]
        self.wind_min = env_cfg["wind_strength_min"]
        self.wind_max = env_cfg["wind_strength_max"]

        self.render_mode = render_mode
        self.window_size = 600
        self.cell_size = self.window_size // self.size

        self.action_space = spaces.MultiDiscrete([5] * self.num_agents)  # 0:right,1:down,2:left,3:up,4:stay

        # Extend observation space with battery level for each agent (battery normalized 0-1)
        self.observation_space = spaces.Tuple([
            spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32) for _ in range(self.num_agents)
        ])

        self.window = None
        self.clock = None

        self._action_to_dir = {
            0: np.array([1, 0]),   # Right
            1: np.array([0, 1]),   # Down
            2: np.array([-1, 0]),  # Left
            3: np.array([0, -1]),  # Up
            4: np.array([0, 0])    # Stay
        }

        # Battery parameters
        self.max_battery = 100.0
        self.battery_drain_per_step = 1.0
        self.battery_drain_per_move = 2.0

        # Track visited cells by all agents (for coverage calculation)
        self.visited_cells = set()
        self.steps = 0
        self.done = False

        self.reset()

    def reset(self, seed=None, options=None):
        self.np_random = np.random.RandomState(seed)

        # Place agents at random distinct positions
        self.agent_positions = []
        taken = set()
        for _ in range(self.num_agents):
            while True:
                pos = tuple(self.np_random.randint(0, self.size, 2))
                if pos not in taken:
                    taken.add(pos)
                    self.agent_positions.append(np.array(pos))
                    break

        # Place targets randomly, distinct from agents
        self.targets = set()
        while len(self.targets) < self.num_targets:
            pos = tuple(self.np_random.randint(0, self.size, 2))
            if pos not in taken:
                self.targets.add(pos)

        # Generate wind map: exactly 10% of the cells have wind
        self.wind_map = {}
        total_cells = self.size * self.size
        num_wind_cells = int(total_cells * self.wind_prob)
        all_positions = [(x, y) for x in range(self.size) for y in range(self.size)]
        wind_positions = self.np_random.choice(len(all_positions), size=num_wind_cells, replace=False)
        for idx in wind_positions:
            x, y = all_positions[idx]
            wind_dir = self._action_to_dir[self.np_random.randint(0, 4)]
            wind_strength = self.np_random.uniform(self.wind_min, self.wind_max)
            self.wind_map[(x, y)] = {"dir": wind_dir, "strength": wind_strength}

        self.visited_cells = set()
        for pos in self.agent_positions:
            self.visited_cells.add(tuple(pos))

        # Initialize battery to full for all agents
        self.batteries = [self.max_battery] * self.num_agents

        self.steps = 0
        self.done = False

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs()

    def _get_obs(self):
        observations = []
        for i, pos in enumerate(self.agent_positions):
            if self.targets:
                targets_np = np.array(list(self.targets))
                dists = np.linalg.norm(targets_np - pos, axis=1)
                nearest_idx = np.argmin(dists)
                nearest_target = targets_np[nearest_idx]
            else:
                nearest_target = pos

            # Observation is position + nearest target + battery (normalized 0-1)
            obs = np.concatenate([
                pos / (self.size - 1),
                nearest_target / (self.size - 1),
                np.array([self.batteries[i] / self.max_battery])
            ]).astype(np.float32)
            observations.append(obs)
        return observations

    def step(self, actions):
        assert len(actions) == self.num_agents
        rewards = [0.0] * self.num_agents

        new_positions = []
        occupied = set()
        for i, (pos, action) in enumerate(zip(self.agent_positions, actions)):
            # If battery is zero, agent cannot move
            if self.batteries[i] <= 0:
                new_positions.append(pos)
                continue

            intended_dir = self._action_to_dir[action]
            next_pos = pos + intended_dir

            wind = self.wind_map.get(tuple(pos))
            if wind:
                strength = wind["strength"]
                if self.np_random.rand() < strength:
                    next_pos = next_pos + wind["dir"]

            next_pos = np.clip(next_pos, 0, self.size - 1)

            # Prevent collision or overlapping positions
            if tuple(next_pos) in occupied or tuple(next_pos) in [tuple(p) for p in self.agent_positions]:
                next_pos = pos

            new_positions.append(next_pos)

        self.agent_positions = new_positions

        # Drain battery per step and additional if moved
        for i in range(self.num_agents):
            # Drain battery per step
            self.batteries[i] -= self.battery_drain_per_step
            # Check if moved
            if not np.array_equal(self.agent_positions[i], self.agent_positions[i]):
                pass  # no movement, no extra drain
            else:
                # Movement occurred if position changed from previous step
                if np.any(self.agent_positions[i] != self.agent_positions[i]):
                    self.batteries[i] -= self.battery_drain_per_move

            # Clamp battery to zero minimum
            self.batteries[i] = max(self.batteries[i], 0.0)

        for i, pos in enumerate(self.agent_positions):
            pos_tuple = tuple(pos)
            if pos_tuple in self.targets:
                rewards[i] += 1.0
                self.targets.remove(pos_tuple)

        # Update visited cells with new positions
        for pos in self.agent_positions:
            self.visited_cells.add(tuple(pos))

        self.steps += 1
        if len(self.targets) == 0 or self.steps >= 200:
            self.done = True

        coverage = len(self.visited_cells) / (self.size * self.size)
        print(f"Step: {self.steps} | Visited cells: {len(self.visited_cells)} / {self.size * self.size} "
              f"({coverage:.2%} coverage)")

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), rewards, self.done, False, {}

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
            pygame.display.set_caption("Multi-UAV Windy Grid World")

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        for x in range(self.size + 1):
            pygame.draw.line(canvas, (200, 200, 200), (x * self.cell_size, 0), (x * self.cell_size, self.window_size))
            pygame.draw.line(canvas, (200, 200, 200), (0, x * self.cell_size), (self.window_size, x * self.cell_size))

        for (x, y), wind in self.wind_map.items():
            start = (x * self.cell_size + self.cell_size // 2, y * self.cell_size + self.cell_size // 2)
            end = (start[0] + int(wind["dir"][0] * 15), start[1] + int(wind["dir"][1] * 15))
            pygame.draw.line(canvas, (0, 128, 0), start, end, 3)
            pygame.draw.circle(canvas, (0, 128, 0), end, 4)

        for (x, y) in self.targets:
            rect = pygame.Rect(x * self.cell_size + 5, y * self.cell_size + 5, self.cell_size - 10, self.cell_size - 10)
            pygame.draw.rect(canvas, (255, 0, 0), rect)

        colors = [(0, 0, 255), (255, 165, 0), (0, 255, 255), (255, 0, 255)]
        for i, pos in enumerate(self.agent_positions):
            center = (int(pos[0] * self.cell_size + self.cell_size / 2), int(pos[1] * self.cell_size + self.cell_size / 2))
            pygame.draw.circle(canvas, colors[i % len(colors)], center, self.cell_size // 3)
            # Draw battery bar above agent
            battery_ratio = self.batteries[i] / self.max_battery
            bar_width = int(self.cell_size * battery_ratio)
            bar_height = 5
            bar_x = center[0] - self.cell_size // 2
            bar_y = center[1] - self.cell_size // 2 - 10
            pygame.draw.rect(canvas, (255, 0, 0), (bar_x, bar_y, self.cell_size, bar_height))
            pygame.draw.rect(canvas, (0, 255, 0), (bar_x, bar_y, bar_width, bar_height))

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window:
            pygame.display.quit()
            pygame.quit()
