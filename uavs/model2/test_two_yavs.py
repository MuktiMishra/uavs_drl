import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from collections import deque
import random
import pandas as pd
import gym
from gym import spaces
import torch.nn as nn
import time # Added for simulation speed control

# Ensure consistent device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Reproducibility (important for consistent testing environments)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Re-define the DiagnosticUAVEnv class (MUST BE IDENTICAL TO TRAINING'S TWO-UAV VERSION)
class DiagnosticUAVEnv(gym.Env):
    def __init__(self, debug=False):
        super(DiagnosticUAVEnv, self).__init__()
        self.debug = debug
        self.grid_size = (15, 15)
        self.grid_height, self.grid_width = self.grid_size

        # Channels: coverage, victim, rescued, obstacle, UAV1, UAV2, battery1, battery2, time_left, risk_map
        self.channels = 10 # Corrected for two UAVs
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.channels, *self.grid_size), dtype=np.float32)
        
        # Action space for two UAVs: 4 actions for UAV1, 4 for UAV2 => 4*4 = 16 discrete actions
        self.action_space = spaces.Discrete(16) # Corrected for two UAVs

        self.max_steps = 400

        self.fixed_victims = [
            (0, 2), (1, 4), (2, 8), (3, 1), (4, 12), (5, 3), (6, 6), (7, 10),
            (8, 14), (9, 7), (10, 5), (11, 2), (12, 13), (13, 0), (14, 4)
        ]
        self.fixed_obstacles = [
            (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 7), (7, 8),
            (8, 9), (9, 10), (10, 11), (11, 12)
        ]

        self.total_victims = len(self.fixed_victims)
        self.base_risk_map = self._create_risk_map()

        self.action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.total_energy_consumption = 0.0
        self.total_risk_score = 0.0

    def _create_risk_map(self):
        risk_map = np.random.rand(self.grid_height, self.grid_width) * 0.2 + 0.1

        for (oy, ox) in self.fixed_obstacles:
            for y in range(max(0, oy-1), min(self.grid_height, oy+2)):
                for x in range(max(0, ox-1), min(self.grid_width, ox+2)):
                    distance = max(abs(y-oy), abs(x-ox))
                    if distance <= 1:
                        risk_map[y, x] += 0.2 * (1 - distance)

        return np.clip(risk_map, 0.1, 0.5)

    def reset(self):
        self.step_count = 0
        self.energy_used1 = 0.0 # Corrected for two UAVs
        self.energy_used2 = 0.0 # Corrected for two UAVs
        self.mission_time = 0.0
        self.risk_score = 0.0
        self.victims_rescued = 0

        self.coverage_map = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        self.victim_map = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        self.rescued_map = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        self.obstacle_map = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)

        valid_positions = [(y, x) for y in range(self.grid_height) for x in range(self.grid_width)
                           if (y, x) not in self.fixed_victims and (y, x) not in self.fixed_obstacles]
        
        # Ensure distinct starting positions for UAVs
        self.uav_pos1 = random.choice(valid_positions) # Corrected for two UAVs
        valid_positions.remove(self.uav_pos1)
        self.uav_pos2 = random.choice(valid_positions) # Corrected for two UAVs

        for (y, x) in self.fixed_victims:
            self.victim_map[y, x] = 1
        for (y, x) in self.fixed_obstacles:
            self.obstacle_map[y, x] = 1

        self.battery1 = 1.0 # Corrected for two UAVs
        self.battery2 = 1.0 # Corrected for two UAVs
        self.time_left = 1.0
        self.risk_map = self.base_risk_map.copy()

        self.position_history1 = deque(maxlen=10) # Corrected for two UAVs
        self.position_history2 = deque(maxlen=10) # Corrected for two UAVs
        self.position_history1.append(self.uav_pos1)
        self.position_history2.append(self.uav_pos2)

        self.total_reward_components = {
            'victim_rescue': 0,
            'exploration': 0,
            'movement': 0,
            'energy_penalty': 0,
            'risk_penalty': 0,
            'stuck_penalty': 0,
            'time_penalty': 0,
            'completion_bonus': 0
        }

        self.total_energy_consumption = 0.0
        self.total_risk_score = 0.0

        return self.get_state()

    def get_state(self):
        uav_layer1 = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        uav_layer1[self.uav_pos1] = 1.0
        uav_layer2 = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        uav_layer2[self.uav_pos2] = 1.0

        battery_layer1 = np.full((self.grid_height, self.grid_width), self.battery1, dtype=np.float32)
        battery_layer2 = np.full((self.grid_height, self.grid_width), self.battery2, dtype=np.float32)
        time_layer = np.full((self.grid_height, self.grid_width), self.time_left, dtype=np.float32)

        coverage_normalized = self.coverage_map.copy()

        stacked = np.stack([
            coverage_normalized,
            self.victim_map,
            self.rescued_map,
            self.obstacle_map,
            uav_layer1,
            uav_layer2,
            battery_layer1,
            battery_layer2,
            time_layer,
            self.risk_map
        ], axis=0)

        return stacked

    def step(self, action):
        self.step_count += 1
        prev_pos1 = self.uav_pos1 # Corrected for two UAVs
        prev_pos2 = self.uav_pos2 # Corrected for two UAVs

        # Decode combined action
        action1 = action // 4 # Corrected for two UAVs
        action2 = action % 4 # Corrected for two UAVs

        # Calculate potential next positions
        new_y1, new_x1 = self.uav_pos1
        if action1 == 0 and new_y1 > 0: new_y1 -= 1
        elif action1 == 1 and new_y1 < self.grid_height - 1: new_y1 += 1
        elif action1 == 2 and new_x1 > 0: new_x1 -= 1
        elif action1 == 3 and new_x1 < self.grid_width - 1: new_x1 += 1
        potential_pos1 = (new_y1, new_x1)

        new_y2, new_x2 = self.uav_pos2
        if action2 == 0 and new_y2 > 0: new_y2 -= 1
        elif action2 == 1 and new_y2 < self.grid_height - 1: new_y2 += 1
        elif action2 == 2 and new_x2 > 0: new_x2 -= 1
        elif action2 == 3 and new_x2 < self.grid_width - 1: new_x2 += 1
        potential_pos2 = (new_y2, new_x2)

        # --- Collision Avoidance Logic (Copied from your two-UAV training code) ---
        uav1_collided_obstacle = (potential_pos1 in self.fixed_obstacles)
        uav2_collided_obstacle = (potential_pos2 in self.fixed_obstacles)
        
        if potential_pos1 == potential_pos2 and potential_pos1 != self.uav_pos1 and potential_pos1 != self.uav_pos2:
            actual_pos1 = self.uav_pos1
            actual_pos2 = self.uav_pos2
        elif potential_pos1 == self.uav_pos2 and potential_pos2 != self.uav_pos1:
            actual_pos1 = self.uav_pos1
            actual_pos2 = potential_pos2 if not uav2_collided_obstacle and potential_pos2 != self.uav_pos1 else self.uav_pos2
        elif potential_pos2 == self.uav_pos1 and potential_pos1 != self.uav_pos2:
            actual_pos2 = self.uav_pos2
            actual_pos1 = potential_pos1 if not uav1_collided_obstacle and potential_pos1 != self.uav_pos2 else self.uav_pos1
        elif potential_pos1 == self.uav_pos2 and potential_pos2 == self.uav_pos1:
            actual_pos1 = self.uav_pos1
            actual_pos2 = self.uav_pos2
        else:
            if not uav1_collided_obstacle and potential_pos1 != potential_pos2:
                actual_pos1 = potential_pos1
            else:
                actual_pos1 = self.uav_pos1

            if not uav2_collided_obstacle and potential_pos2 != actual_pos1:
                actual_pos2 = potential_pos2
            else:
                actual_pos2 = self.uav_pos2
        
        self.uav_pos1 = actual_pos1 # Corrected for two UAVs
        self.uav_pos2 = actual_pos2 # Corrected for two UAVs
        # --- End Collision Avoidance Logic ---

        self.position_history1.append(self.uav_pos1) # Corrected for two UAVs
        self.position_history2.append(self.uav_pos2) # Corrected for two UAVs

        was_new_area1 = self.coverage_map[self.uav_pos1] == 0
        was_new_area2 = self.coverage_map[self.uav_pos2] == 0
        was_new_area = was_new_area1 or was_new_area2
        self.coverage_map[self.uav_pos1] = 1.0
        self.coverage_map[self.uav_pos2] = 1.0

        victim_rescued = False
        if self.victim_map[self.uav_pos1] == 1 and self.rescued_map[self.uav_pos1] == 0:
            self.rescued_map[self.uav_pos1] = 1
            self.victims_rescued += 1
            victim_rescued = True
        if self.victim_map[self.uav_pos2] == 1 and self.rescued_map[self.uav_pos2] == 0:
            self.rescued_map[self.uav_pos2] = 1
            self.victims_rescued += 1
            victim_rescued = True

        energy_consumed1 = 0.003
        if self.uav_pos1 != prev_pos1: # Moved
            energy_consumed1 += 0.002
        if self.uav_pos1 in self.fixed_victims and self.rescued_map[self.uav_pos1] == 1 and prev_pos1 != self.uav_pos1: # Rescued this step
            energy_consumed1 += 0.005
        
        energy_consumed2 = 0.003
        if self.uav_pos2 != prev_pos2: # Moved
            energy_consumed2 += 0.002
        if self.uav_pos2 in self.fixed_victims and self.rescued_map[self.uav_pos2] == 1 and prev_pos2 != self.uav_pos2: # Rescued this step
            energy_consumed2 += 0.005

        self.energy_used1 += energy_consumed1 # Corrected for two UAVs
        self.battery1 = max(0, self.battery1 - energy_consumed1) # Corrected for two UAVs
        self.energy_used2 += energy_consumed2 # Corrected for two UAVs
        self.battery2 = max(0, self.battery2 - energy_consumed2) # Corrected for two UAVs
        
        self.total_energy_consumption += (energy_consumed1 + energy_consumed2) # Corrected for two UAVs


        time_step = 1.0 / self.max_steps
        self.mission_time += time_step
        self.time_left = max(0, self.time_left - time_step)

        self.risk_score += (self.risk_map[self.uav_pos1] * 0.01 + self.risk_map[self.uav_pos2] * 0.01) # Corrected for two UAVs
        self.total_risk_score += (self.risk_map[self.uav_pos1] * 0.01 + self.risk_map[self.uav_pos2] * 0.01) # Corrected for two UAVs


        done = (self.step_count >= self.max_steps or
                (self.battery1 <= 0 and self.battery2 <= 0) or # Corrected for two UAVs
                self.time_left <= 0 or
                self.victims_rescued == self.total_victims)

        reward = self._calculate_detailed_reward(victim_rescued, prev_pos1, prev_pos2, was_new_area) # Corrected for two UAVs

        next_state = self.get_state()

        info = {
            'energy_used1': self.energy_used1, # Corrected for two UAVs
            'energy_used2': self.energy_used2, # Corrected for two UAVs
            'mission_time': self.mission_time,
            'risk_score': self.risk_score,
            'battery1': self.battery1, # Corrected for two UAVs
            'battery2': self.battery2, # Corrected for two UAVs
            'time_left': self.time_left,
            'step': self.step_count,
            'victims_rescued': self.victims_rescued,
            'total_victims': self.total_victims,
            'victim_rescued_this_step': victim_rescued,
            'coverage_ratio': np.sum(self.coverage_map) / (self.grid_height * self.grid_width),
            'reward_components': self.total_reward_components.copy(),
            'action_taken_uav1': self.action_names[action1], # Corrected for two UAVs
            'action_taken_uav2': self.action_names[action2], # Corrected for two UAVs
            'position_uav1': self.uav_pos1, # Corrected for two UAVs
            'position_uav2': self.uav_pos2, # Corrected for two UAVs
            'total_energy_consumption': self.total_energy_consumption,
            'cumulative_risk_score': self.total_risk_score
        }

        return next_state, reward, done, info

    def _calculate_detailed_reward(self, victim_rescued, prev_pos1, prev_pos2, was_new_area): # Corrected args
        reward_components = {
            'victim_rescue': 0,
            'exploration': 0,
            'movement': 0,
            'energy_penalty': 0,
            'risk_penalty': 0,
            'stuck_penalty': 0,
            'time_penalty': 0,
            'completion_bonus': 0
        }

        if victim_rescued:
            reward_components['victim_rescue'] = 25.0

        if was_new_area:
            reward_components['exploration'] = 2.0

        # Movement for each UAV
        if self.uav_pos1 != prev_pos1:
            reward_components['movement'] += 0.2
        else:
            reward_components['movement'] -= 0.2
        if self.uav_pos2 != prev_pos2:
            reward_components['movement'] += 0.2
        else:
            reward_components['movement'] -= 0.2
        
        # Combined energy penalty
        reward_components['energy_penalty'] = -(self.energy_used1 + self.energy_used2) * 0.4 # Corrected

        # Combined risk penalty
        reward_components['risk_penalty'] = -(self.risk_map[self.uav_pos1] * 0.15 + self.risk_map[self.uav_pos2] * 0.15) # Corrected

        # Stuck penalty if both UAVs are stuck
        are_uav1_stuck = len(set(self.position_history1)) <= 3
        are_uav2_stuck = len(set(self.position_history2)) <= 3
        if are_uav1_stuck and are_uav2_stuck and self.step_count > self.max_steps * 0.1:
            reward_components['stuck_penalty'] = -2.0

        reward_components['time_penalty'] = -0.02

        if self.victims_rescued == self.total_victims:
            efficiency_bonus = ((self.battery1 + self.battery2) / 2 * 20.0 + self.time_left * 20.0) # Corrected
            reward_components['completion_bonus'] = 100.0 + efficiency_bonus

        for key, value in reward_components.items():
            self.total_reward_components[key] += value

        total_reward = sum(reward_components.values())
        return total_reward

    def print_debug_info(self):
        print(f"Episode completed in {self.step_count} steps")
        print(f"Victims rescued: {self.victims_rescued}/{self.total_victims}")
        print(f"Coverage: {np.sum(self.coverage_map)}/{self.grid_height * self.grid_width} ({np.sum(self.coverage_map)/(self.grid_height * self.grid_width)*100:.1f}%)")
        print(f"UAV1 final position: {self.uav_pos1}") # Corrected
        print(f"UAV2 final position: {self.uav_pos2}") # Corrected
        print(f"Battery1 remaining: {self.battery1:.3f}") # Corrected
        print(f"Battery2 remaining: {self.battery2:.3f}") # Corrected
        print(f"Time remaining: {self.time_left:.3f}")
        print("Reward components:")
        for component, value in self.total_reward_components.items():
            if abs(value) > 0.01:
                print(f"   {component}: {value:.2f}")
        print()

# Re-define the DiagnosticCNNActorCritic class (MUST BE IDENTICAL TO TRAINING'S TWO-UAV VERSION)
class DiagnosticCNNActorCritic(nn.Module):
    def __init__(self, input_channels=10, num_actions=16): # Corrected input_channels and num_actions
        super(DiagnosticCNNActorCritic, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 15 * 15, 512)
        self.fc2 = nn.Linear(512, 256)

        self.policy_head = nn.Linear(256, num_actions)
        self.value_head = nn.Linear(256, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        if self.policy_head.bias is not None:
            nn.init.constant_(self.policy_head.bias, 0)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        if self.value_head.bias is not None:
            nn.init.constant_(self.value_head.bias, 0)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        policy_logits = self.policy_head(x)
        value = self.value_head(x)

        return policy_logits, value

def visualize_episode(env_instance, path_history1, path_history2, episode_num, total_reward, victims_rescued, 
                      grid_size, fixed_obstacles, fixed_victims, rescued_victims_set,
                      final_battery1, final_battery2, final_time_left, final_risk_score):
    fig, ax = plt.subplots(figsize=(grid_size[1], grid_size[0]))

    # Default: White (uncovered)
    grid_display = np.ones(grid_size + (3,), dtype=np.float32)

    # Covered cells: Grey
    covered_cells_mask = env_instance.coverage_map == 1
    grid_display[covered_cells_mask] = [0.7, 0.7, 0.7] # Light grey

    # Obstacles: Black
    obstacle_cells_mask = env_instance.obstacle_map == 1
    grid_display[obstacle_cells_mask] = [0, 0, 0] # Black

    # Victims: Red (if not rescued)
    unrescued_victims_mask = (env_instance.victim_map == 1) & (env_instance.rescued_map == 0)
    grid_display[unrescued_victims_mask] = [1, 0, 0] # Red

    ax.imshow(grid_display)
    ax.set_xticks(np.arange(-0.5, grid_size[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_size[0], 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    ax.set_aspect('equal')
    ax.set_xlim(-0.5, grid_size[1] - 0.5)
    ax.set_ylim(-0.5, grid_size[0] - 0.5)
    ax.invert_yaxis()

    # Plot UAV paths (optional, can make plot busy)
    # path_y1 = [pos[0] for pos in path_history1]
    # path_x1 = [pos[1] for pos in path_history1]
    # ax.plot(path_x1, path_y1, 'b:', markersize=0, linewidth=1, alpha=0.5, label='UAV1 Path')

    # path_y2 = [pos[0] for pos in path_history2]
    # path_x2 = [pos[1] for pos in path_history2]
    # ax.plot(path_x2, path_y2, 'g:', markersize=0, linewidth=1, alpha=0.5, label='UAV2 Path')

    # Plot current UAV positions as triangles
    y1, x1 = path_history1[-1]
    uav1_triangle = patches.RegularPolygon((x1, y1), numVertices=3, radius=0.4, 
                                           orientation=np.pi/2, facecolor='blue', edgecolor='black', lw=1, label='UAV1')
    ax.add_patch(uav1_triangle)

    y2, x2 = path_history2[-1]
    uav2_triangle = patches.RegularPolygon((x2, y2), numVertices=3, radius=0.4, 
                                           orientation=np.pi/2, facecolor='green', edgecolor='black', lw=1, label='UAV2')
    ax.add_patch(uav2_triangle)

    # Add text for UAVs
    ax.text(x1, y1 - 0.4, 'UAV1', color='white', ha='center', va='bottom', fontsize=8, backgroundcolor='blue', alpha=0.7)
    ax.text(x2, y2 - 0.4, 'UAV2', color='white', ha='center', va='bottom', fontsize=8, backgroundcolor='green', alpha=0.7)


    ax.set_title(f"Test Episode {episode_num} - Reward: {total_reward:.2f}, Victims: {victims_rescued}/{env_instance.total_victims}\n"
                 f"Battery1: {final_battery1:.2f}, Battery2: {final_battery2:.2f} | Time Left: {final_time_left:.2f} | Risk: {final_risk_score:.2f}")

    # Create a dummy legend for clarity
    handles = [
        patches.Patch(color='black', label='Obstacle'),
        patches.Patch(color='red', label='Unrescued Victim'),
        patches.Patch(color='green', label='Rescued Victim'),
        patches.Patch(color=[0.7, 0.7, 0.7], label='Covered Cell'),
        patches.Patch(color='white', label='Uncovered Cell'),
        patches.RegularPolygon((0,0), numVertices=3, radius=0.1, facecolor='blue', edgecolor='black', label='UAV1'),
        patches.RegularPolygon((0,0), numVertices=3, radius=0.1, facecolor='green', edgecolor='black', label='UAV2')
    ]
    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
    plot_dir = "model_output/test_plots"
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, f"test_episode_{episode_num:03d}.png"))
    plt.close()

def test_model(model_path="model_output/two_uavs.pth", num_test_episodes=5): # Corrected default model path
    env = DiagnosticUAVEnv(debug=False)
    model = DiagnosticCNNActorCritic(input_channels=env.channels, num_actions=env.action_space.n).to(device)

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please train the model first by running the main training script.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Set model to evaluation mode

    test_results = []

    print(f"\n--- Starting {num_test_episodes} test episodes ---")

    for i in range(num_test_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        path_history1 = []
        path_history2 = []
        rescued_victims_this_episode = set() # To track actual rescued cells for plotting

        # Store initial positions
        path_history1.append(env.uav_pos1)
        path_history2.append(env.uav_pos2)

        print(f"Test Episode {i+1}/{num_test_episodes}")

        while not done:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                logits, _ = model(state_tensor)
                probs = torch.softmax(logits, dim=1)
                action = torch.argmax(probs, dim=1).item() # Greedily choose action during testing

            next_state, reward, done, info = env.step(action)

            episode_reward += reward
            state = next_state
            path_history1.append(info['position_uav1'])
            path_history2.append(info['position_uav2'])

            # Update rescued victims set based on the info from the environment
            # Note: The 'info' dictionary provides a 'victim_rescued_this_step' boolean
            # but doesn't directly give the *location* if it was rescued.
            # We can infer it from the `rescued_map` after the step.
            for r_y in range(env.grid_height):
                for r_x in range(env.grid_width):
                    if env.rescued_map[r_y, r_x] == 1 and (r_y, r_x) in env.fixed_victims:
                        rescued_victims_this_episode.add((r_y, r_x))


        env.print_debug_info() # Print detailed info for each test episode

        visualize_episode(env, path_history1, path_history2, i+1, episode_reward, info['victims_rescued'], 
                          env.grid_size, env.fixed_obstacles, env.fixed_victims, rescued_victims_this_episode,
                          info['battery1'], info['battery2'], info['time_left'], info['risk_score'])

        test_results.append({
            'episode': i + 1,
            'total_reward': episode_reward,
            'victims_rescued': info['victims_rescued'],
            'coverage_ratio': info['coverage_ratio'],
            'steps_taken': info['step'],
            'battery1_final': info['battery1'], # Corrected
            'battery2_final': info['battery2'], # Corrected
            'time_left_final': info['time_left'],
            'risk_score_final': info['risk_score']
        })

    results_df = pd.DataFrame(test_results)
    results_csv_path = "model_output/test_results_two_uavs.csv" # Changed CSV name
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nTest results saved to {results_csv_path}")
    print("\n--- Testing complete ---")

    # Overall average results
    print("\nOverall Average Test Results:")
    print(f"   Avg Total Reward: {results_df['total_reward'].mean():.2f}")
    print(f"   Avg Victims Rescued: {results_df['victims_rescued'].mean():.2f}/{env.total_victims}")
    print(f"   Avg Coverage Ratio: {results_df['coverage_ratio'].mean():.3f}")
    print(f"   Avg Steps Taken: {results_df['steps_taken'].mean():.1f}")
    print(f"   Avg Final Battery 1: {results_df['battery1_final'].mean():.3f}") # Corrected
    print(f"   Avg Final Battery 2: {results_df['battery2_final'].mean():.3f}") # Corrected
    print(f"   Avg Final Time Left: {results_df['time_left_final'].mean():.3f}")
    print(f"   Avg Final Risk Score: {results_df['risk_score_final'].mean():.3f}")


if __name__ == "__main__":
    # Specify the path to your trained model
    model_to_test = "model_output/two_uavs.pth" # Corrected model path for two UAVs

    if not os.path.exists(model_to_test):
        print(f"No trained model found at {model_to_test}.")
        print("Please ensure you have run the training script and the model 'two_uavs.pth' exists.")
        exit()

    # Number of episodes to run for testing
    num_episodes_to_test = 5

    test_model(model_path=model_to_test, num_test_episodes=num_episodes_to_test)