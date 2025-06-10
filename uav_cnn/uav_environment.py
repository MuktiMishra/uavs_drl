# import gym
# from gym import spaces
# import torch
# import numpy as np
# import random

# class UAVEnvironment(gym.Env):
#     def __init__(self, height=15, width=15, init_battery=1.0, total_time=1.0, source=(0, 0)):
#         super().__init__()
#         self.H = height
#         self.W = width
#         self.source = source
#         self.init_battery = init_battery
#         self.battery = init_battery
#         self.total_time = total_time
#         self.time_left = total_time
        
#         #Action space: 4 low actions as of now later total 8 (4 high level action spaces)
#         self.action_space = spaces.Discrete(4)
        
#         #Observation: same as before (7, H, W)
#         self.observation_space = spaces.Box(low=0, high=1, shape=(7, self.H, self.W), dtype=np.float32)
        
#         #initializing  maps
#         self.coverage_map = torch.zeros((self.H, self.W))
#         self.victim_map = torch.zeros((self.H, self.W))
#         self.risk_map = torch.rand((self.H, self.W)) * 0.5
#         self.obstacle_map = torch.zeros((self.H, self.W))
#         self.position_map = torch.zeros((self.H, self.W))
#         self.battery_map = -1 * torch.ones((self.H, self.W))
#         self.time_map = -1 * torch.ones((self.H, self.W))
        
#         # Wind map: speed and direction per cell
#         # Wind speed v in [0,1], direction encoded as 0:UP,1:DOWN,2:LEFT,3:RIGHT
#         self.wind_speed = torch.rand((self.H, self.W))
#         self.wind_dir = torch.randint(0, 4, (self.H, self.W))
        
#         self.uav_pos = self.source
        
#         # Battery parameters
#         self.base_energy_per_action = 0.005  
#         self.battery_degradation = 0.001     
#         self.env_noise_mean = 0
#         self.env_noise_std = 0.0012           
        
#         self.task_energy_costs = {0: 0.002, 1: 0.01, 2: 0.015, 3: 0.005}  
        
#         self.k_wind = 0.03  # wind energy cost factor
        
#         self.reset()
        
#     def reset(self):
#         self.coverage_map.fill_(0)
#         self.victim_map.fill_(0)
#         self.risk_map = torch.rand((self.H, self.W)) * 0.5
#         self.obstacle_map.fill_(0)
#         self.position_map.fill_(0)
#         self.battery_map.fill_(-1)
#         self.time_map.fill_(-1)
        
#         self.uav_pos = self.source
#         self.battery = self.init_battery
#         self.time_left = self.total_time
        
#         self.set_uav_initial_position()
#         return self.get_state()
    
#     def set_uav_initial_position(self):
#         x, y = self.uav_pos
#         self.position_map.fill_(0)
#         self.position_map[x, y] = 1
#         self.coverage_map[x, y] = 1
#         self.battery_map[x, y] = 1.0
#         self.time_map[x, y] = 1.0
    
#     def get_state(self):
#         state = torch.stack([
#             self.coverage_map,
#             self.victim_map,
#             self.risk_map,
#             self.obstacle_map,
#             self.position_map,
#             self.battery_map,
#             self.time_map
#         ]).numpy().astype(np.float32)
#         return state
    
#     def step(self, action):
#         #movement for action a₁ and a₄ (for simplicity, say reroute = random move)
#         move_dict = {
#             0: (0, 1),   #explore Path: move right for demo
#             3: (random.choice([-1,0,1]), random.choice([-1,0,1])),  
#         }
        
#         x, y = self.uav_pos
        
#         if action == 0 or action == 3:  # Move actions
#             dx, dy = move_dict[action]
#             new_x, new_y = x + dx, y + dy
            
#             # Check boundaries
#             if not (0 <= new_x < self.H and 0 <= new_y < self.W):
#                 new_x, new_y = x, y  # no move
            
#             # Calculate wind effect energy cost
#             wind_v = self.wind_speed[new_x, new_y].item()
#             wind_theta = self.wind_dir[new_x, new_y].item()
            
#             # Calculate alignment between move and wind direction
#             move_dir = self.get_direction(dx, dy)
#             alignment_factor = self.calculate_wind_alignment(move_dir, wind_theta)
            
#             wind_energy_cost = self.k_wind * wind_v * alignment_factor
            
#             # Total energy consumption
#             env_noise = random.gauss(self.env_noise_mean, self.env_noise_std)
#             task_energy = self.task_energy_costs.get(action, 0.005)
            
#             delta_battery = self.base_energy_per_action + self.battery_degradation + wind_energy_cost + task_energy + env_noise
#             self.battery = max(0, self.battery - delta_battery)
#             self.time_left = max(0, self.time_left - 0.01)
            
#             # Update position
#             self.uav_pos = (new_x, new_y)
#             self.update_maps(new_x, new_y)
            
#             # Reward example: encourage coverage, penalize risk
#             reward = 1 - self.risk_map[new_x, new_y].item()
#             done = self.battery <= 0 or self.time_left <= 0
            
#             return self.get_state(), reward, done, {}
        
#         elif action == 1:
#             # Task allocation logic here
#             # Could mean switching tasks, cost more battery
#             self.battery = max(0, self.battery - 0.01)
#             reward = 0.5  # example
#             done = False
#             return self.get_state(), reward, done, {}
        
#         elif action == 2:
#             # Task handover logic here
#             self.battery = max(0, self.battery - 0.02)
#             reward = -0.1  # cost for handover
#             done = False
#             return self.get_state(), reward, done, {}
    
#     def get_direction(self, dx, dy):
#         # Map dx,dy to direction id (0:UP,1:DOWN,2:LEFT,3:RIGHT)
#         if dx == -1 and dy == 0:
#             return 0
#         elif dx == 1 and dy == 0:
#             return 1
#         elif dx == 0 and dy == -1:
#             return 2
#         elif dx == 0 and dy == 1:
#             return 3
#         else:
#             return -1  # no movement or diagonal
    
#     def calculate_wind_alignment(self, move_dir, wind_dir):
#         # Return factor based on alignment
#         if move_dir == -1:
#             return 1  # no movement, normal cost
#         if move_dir == wind_dir:
#             return 1  # aligned, normal cost
#         # opposite directions
#         if (move_dir == 0 and wind_dir == 1) or (move_dir == 1 and wind_dir == 0) or \
#            (move_dir == 2 and wind_dir == 3) or (move_dir == 3 and wind_dir == 2):
#             return 2  # double cost for resistance
#         # perpendicular
#         return 1.5  # intermediate cost
    
#     def update_maps(self, x, y):
#         self.position_map.fill_(0)
#         self.position_map[x, y] = 1
#         self.coverage_map[x, y] = 1
#         self.battery_map[x, y] = self.battery
#         self.time_map[x, y] = self.time_left
    
#     def reset(self):
#         self.coverage_map.fill_(0)
#         self.victim_map.fill_(0)
#         self.risk_map = torch.rand((self.H, self.W)) * 0.5
#         self.obstacle_map.fill_(0)
#         self.position_map.fill_(0)
#         self.battery_map.fill_(-1)
#         self.time_map.fill_(-1)
    
#         self.uav_pos = self.source
#         self.battery = self.init_battery
#         self.time_left = self.total_time
    
#         self.set_uav_initial_position()
#         return self.get_state()


# (Same import section)
import gym
from gym import spaces
import torch
import numpy as np
import random

class UAVEnvironment(gym.Env):
    def __init__(self, height=15, width=15, init_battery=1.0, total_time=1.0, source=(0, 0)):
        super().__init__()
        self.H = height
        self.W = width
        self.source = source
        self.init_battery = init_battery
        self.battery = init_battery
        self.total_time = total_time
        self.time_left = total_time

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1, shape=(7, self.H, self.W), dtype=np.float32)

        self.coverage_map = torch.zeros((self.H, self.W))
        self.victim_map = torch.zeros((self.H, self.W))
        self.risk_map = torch.rand((self.H, self.W)) * 0.5
        self.obstacle_map = torch.zeros((self.H, self.W))
        self.position_map = torch.zeros((self.H, self.W))
        self.battery_map = -1 * torch.ones((self.H, self.W))
        self.time_map = -1 * torch.ones((self.H, self.W))

        self.wind_speed = torch.rand((self.H, self.W))
        self.wind_dir = torch.randint(0, 4, (self.H, self.W))

        self.base_energy_per_action = 0.005  
        self.battery_degradation = 0.001     
        self.env_noise_mean = 0
        self.env_noise_std = 0.0012           
        self.task_energy_costs = {0: 0.002, 1: 0.01, 2: 0.015, 3: 0.005}  
        self.k_wind = 0.03  

        self.victims = [(3, 4), (7, 8), (10, 2)]  # fixed victims
        self.reached_victims = set()

        self.uav_pos = self.source
        self.reset()

    def reset(self):
        self.coverage_map.fill_(0)
        self.victim_map.fill_(0)
        for vx, vy in self.victims:
            self.victim_map[vx, vy] = 1
        self.risk_map = torch.rand((self.H, self.W)) * 0.5
        self.obstacle_map.fill_(0)
        self.position_map.fill_(0)
        self.battery_map.fill_(-1)
        self.time_map.fill_(-1)

        self.uav_pos = self.source
        self.battery = self.init_battery
        self.time_left = self.total_time
        self.reached_victims = set()

        self.set_uav_initial_position()
        return self.get_state()

    def set_uav_initial_position(self):
        x, y = self.uav_pos
        self.position_map.fill_(0)
        self.position_map[x, y] = 1
        self.coverage_map[x, y] = 1
        self.battery_map[x, y] = 1.0
        self.time_map[x, y] = 1.0

    def get_state(self):
        return torch.stack([
            self.coverage_map,
            self.victim_map,
            self.risk_map,
            self.obstacle_map,
            self.position_map,
            self.battery_map,
            self.time_map
        ]).numpy().astype(np.float32)

    def step(self, action):
        move_dict = {
            0: (0, 1),   # move right
            3: (random.choice([-1, 0, 1]), random.choice([-1, 0, 1]))  # random move
        }

        x, y = self.uav_pos
        dx, dy = move_dict.get(action, (0, 0))
        new_x, new_y = x + dx, y + dy

        # Boundary check
        if not (0 <= new_x < self.H and 0 <= new_y < self.W):
            new_x, new_y = x, y

        wind_v = self.wind_speed[new_x, new_y].item()
        wind_theta = self.wind_dir[new_x, new_y].item()
        move_dir = self.get_direction(dx, dy)
        alignment_factor = self.calculate_wind_alignment(move_dir, wind_theta)
        wind_energy_cost = self.k_wind * wind_v * alignment_factor
        env_noise = random.gauss(self.env_noise_mean, self.env_noise_std)
        task_energy = self.task_energy_costs.get(action, 0.005)

        delta_battery = self.base_energy_per_action + self.battery_degradation + wind_energy_cost + task_energy + env_noise
        self.battery = max(0, self.battery - delta_battery)
        self.time_left = max(0, self.time_left - 0.01)

        self.uav_pos = (new_x, new_y)
        self.update_maps(new_x, new_y)

        # ---- REWARD ----
        reward = 0.0
        if self.coverage_map[new_x, new_y] == 0:
            reward += 1 - self.risk_map[new_x, new_y].item()

        if (new_x, new_y) in self.victims and (new_x, new_y) not in self.reached_victims:
            self.reached_victims.add((new_x, new_y))
            reward += 5.0

        done = self.battery <= 0 or self.time_left <= 0 or len(self.reached_victims) == len(self.victims)
        return self.get_state(), reward, done, {}

    def get_direction(self, dx, dy):
        if dx == -1 and dy == 0: return 0
        if dx == 1 and dy == 0: return 1
        if dx == 0 and dy == -1: return 2
        if dx == 0 and dy == 1: return 3
        return -1

    def calculate_wind_alignment(self, move_dir, wind_dir):
        if move_dir == -1 or move_dir == wind_dir: return 1
        if (move_dir, wind_dir) in [(0, 1), (1, 0), (2, 3), (3, 2)]: return 2
        return 1.5

    def update_maps(self, x, y):
        self.position_map.fill_(0)
        self.position_map[x, y] = 1
        self.coverage_map[x, y] = 1
        self.battery_map[x, y] = self.battery
        self.time_map[x, y] = self.time_left
