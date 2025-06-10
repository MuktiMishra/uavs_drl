import gym
import numpy as np
import importlib.util
import sys
import os

# Load .config.py dynamically
config_path = os.path.join(os.path.dirname(__file__), '.config.py')
spec = importlib.util.spec_from_file_location("config", config_path)
config_module = importlib.util.module_from_spec(spec)
sys.modules["config"] = config_module
spec.loader.exec_module(config_module)

Config = config_module.Config


class SARGridEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.grid_h = Config.GRID_H
        self.grid_w = Config.GRID_W
        self.channels = Config.CHANNELS

        self.observation_space = gym.spaces.Box(
            low=0, high=1,
            shape=(self.channels, self.grid_h, self.grid_w),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(Config.NUM_ACTIONS)
        self.reset()

    def reset(self):
        self.coverage_map = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        self.victim_map = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        self.risk_map = np.random.uniform(0, 0.5, size=(self.grid_h, self.grid_w)).astype(np.float32)
        self.obstacle_map = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        self.uav_pos = [self.grid_h // 2, self.grid_w // 2]
        self.battery = 1.0
        self.mission_time = 1.0

        num_victims = int(0.1 * self.grid_h * self.grid_w)
        victim_indices = np.random.choice(self.grid_h * self.grid_w, num_victims, replace=False)
        for idx in victim_indices:
            x, y = divmod(idx, self.grid_w)
            self.victim_map[x, y] = 1.0

        num_obstacles = int(0.1 * self.grid_h * self.grid_w)
        obstacle_indices = np.random.choice(self.grid_h * self.grid_w, num_obstacles, replace=False)
        for idx in obstacle_indices:
            x, y = divmod(idx, self.grid_w)
            if (x, y) == tuple(self.uav_pos) or self.victim_map[x, y] == 1:
                continue
            self.obstacle_map[x, y] = 1.0

        self.wind_speed = np.random.uniform(0, 1, size=(self.grid_h, self.grid_w)).astype(np.float32)
        self.wind_dir = np.random.randint(0, 4, size=(self.grid_h, self.grid_w))
        self.timestep = 0
        self.max_timesteps = 200

        self.update_state()
        return self.state

    def update_state(self):
        pos_map = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        pos_map[self.uav_pos[0], self.uav_pos[1]] = 1.0
        battery_map = np.full((self.grid_h, self.grid_w), self.battery, dtype=np.float32)
        time_map = np.full((self.grid_h, self.grid_w), self.mission_time, dtype=np.float32)

        self.state = np.stack([
            self.coverage_map,
            self.victim_map,
            self.risk_map,
            self.obstacle_map,
            pos_map,
            battery_map,
            time_map,
        ], axis=0).astype(np.float32)
    
    def move(self, pos, action):
        x, y = pos
        if action == 0 and x > 0:
            x -= 1        # Move Up
        elif action == 1 and y < self.grid_w - 1:
            y += 1        # Move Right
        elif action == 2 and x < self.grid_h - 1:
            x += 1        # Move Down
        elif action == 3 and y > 0:
            y -= 1        # Move Left
        return [x, y]


    # def step(self, action):
    #     self.timestep += 1
    #     reward = 0.0
    #     done = False

    #     current_pos = self.uav_pos.copy()
    #     new_pos = current_pos.copy()
    #     direction = self.timestep % 4

    #     def move(pos, dir):
    #         x, y = pos
    #         if dir == 0 and x > 0:
    #             x -= 1
    #         elif dir == 1 and y < self.grid_w - 1:
    #             y += 1
    #         elif dir == 2 and x < self.grid_h - 1:
    #             x += 1
    #         elif dir == 3 and y > 0:
    #             y -= 1
    #         return [x, y]

    #     if action == 0:
    #         new_pos = move(current_pos, direction)
    #     elif action == 1:
    #         new_pos = self.move_toward_nearest_victim(current_pos)
    #     elif action == 2:
    #         new_pos = current_pos
    #     elif action == 3:
    #         for dir_try in range(4):
    #             candidate = move(current_pos, (direction + dir_try) % 4)
    #             if self.obstacle_map[candidate[0], candidate[1]] == 0:
    #                 new_pos = candidate
    #                 break

    #     if self.obstacle_map[new_pos[0], new_pos[1]] == 1:
    #         new_pos = current_pos

    #     self.uav_pos = new_pos
    #     self.coverage_map[self.uav_pos[0], self.uav_pos[1]] = 1.0

    #     base_cost = Config.BASE_ENERGY
    #     action_cost = Config.ENERGY_ACTION_COSTS[action]
    #     task_cost = Config.TASK_ENERGY if self.victim_map[self.uav_pos[0], self.uav_pos[1]] == 1 else 0.0
    #     noise = np.random.normal(0, Config.ENERGY_NOISE_STD)
    #     wind_cost = self.calc_wind_cost(current_pos, new_pos)

    #     energy_consumed = base_cost + action_cost + task_cost + noise + wind_cost
    #     self.battery = max(0.0, self.battery - energy_consumed)

    #     if self.victim_map[self.uav_pos[0], self.uav_pos[1]] == 1:
    #         reward += 10.0
    #         self.victim_map[self.uav_pos[0], self.uav_pos[1]] = 0.0

    #     reward -= energy_consumed
    #     self.mission_time = max(0.0, 1.0 - self.timestep / self.max_timesteps)
    #     done = (self.battery <= 0) or (self.mission_time <= 0) or (self.victim_map.sum() == 0)

    #     self.update_state()
    #     info = {}
    #     return self.state, reward, done, info

    def step(self, action):
        self.timestep += 1
        reward = 0.0
        done = False
    
        current_pos = self.uav_pos.copy()
    
        if action in [0, 1, 2, 3]:
            new_pos = self.move(current_pos, action)
        elif action == 4:  # If you want an action to move toward nearest victim
            new_pos = self.move_toward_nearest_victim(current_pos)
        elif action == 5:  # Or 'stay put' action if you want
            new_pos = current_pos
        else:
            new_pos = current_pos  # fallback, no move
    
        # Prevent moving into obstacles
        if self.obstacle_map[new_pos[0], new_pos[1]] == 1:
            new_pos = current_pos
    
        self.uav_pos = new_pos
        self.coverage_map[self.uav_pos[0], self.uav_pos[1]] = 1.0
    
        # Energy and reward calculation unchanged
        base_cost = Config.BASE_ENERGY
        action_cost = Config.ENERGY_ACTION_COSTS[action] if action < len(Config.ENERGY_ACTION_COSTS) else 0.0
        task_cost = Config.TASK_ENERGY if self.victim_map[self.uav_pos[0], self.uav_pos[1]] == 1 else 0.0
        noise = np.random.normal(0, Config.ENERGY_NOISE_STD)
        wind_cost = self.calc_wind_cost(current_pos, new_pos)
    
        energy_consumed = base_cost + action_cost + task_cost + noise + wind_cost
        self.battery = max(0.0, self.battery - energy_consumed)
    
        if self.victim_map[self.uav_pos[0], self.uav_pos[1]] == 1:
            reward += 10.0
            self.victim_map[self.uav_pos[0], self.uav_pos[1]] = 0.0
    
        reward -= energy_consumed
        self.mission_time = max(0.0, 1.0 - self.timestep / self.max_timesteps)
        done = (self.battery <= 0) or (self.mission_time <= 0) or (self.victim_map.sum() == 0)
    
        self.update_state()
        info = {}
        return self.state, reward, done, info


    def move_toward_nearest_victim(self, pos):
        from collections import deque
        queue = deque([pos])
        visited = set([tuple(pos)])
        while queue:
            cur = queue.popleft()
            if self.victim_map[cur[0], cur[1]] == 1:
                return self.step_toward(pos, cur)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cur[0] + dx, cur[1] + dy
                if 0 <= nx < self.grid_h and 0 <= ny < self.grid_w and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append([nx, ny])
        return pos

    def step_toward(self, start, goal):
        x0, y0 = start
        x1, y1 = goal
        if x1 < x0 and self.obstacle_map[x0 - 1, y0] == 0:
            return [x0 - 1, y0]
        elif x1 > x0 and self.obstacle_map[x0 + 1, y0] == 0:
            return [x0 + 1, y0]
        elif y1 < y0 and self.obstacle_map[x0, y0 - 1] == 0:
            return [x0, y0 - 1]
        elif y1 > y0 and self.obstacle_map[x0, y0 + 1] == 0:
            return [x0, y0 + 1]
        return start

    def calc_wind_cost(self, old_pos, new_pos):
        if old_pos == new_pos:
            return 0.0
        x0, y0 = old_pos
        x1, y1 = new_pos

        if x1 < x0:
            d = 0
        elif x1 > x0:
            d = 2
        elif y1 > y0:
            d = 3
        elif y1 < y0:
            d = 1
        else:
            return 0.0

        v = self.wind_speed[x1, y1]
        theta = self.wind_dir[x1, y1]

        if d == theta:
            align_factor = 0.5
        elif (d + 2) % 4 == theta:
            align_factor = 1.0
        else:
            align_factor = 0.75

        return Config.WIND_K * v * align_factor
