import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import os
import gym
from gym import spaces
from collections import deque
import matplotlib.pyplot as plt
import math


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


ACTION_ENERGY_COSTS = {
    0: 0.002, # Explore Path
    1: 0.001, # Task Allocation
    2: 0.001, # Task Handover
    3: 0.003  # Reroute
}
TASK_ENERGY_RESCUE = 0.005 # Extra energy for rescuing a victim
BASE_ENERGY_STEP = 0.002 # Base energy cost for any step
ENV_NOISE_STD = 0.001 # Standard deviation for environmental noise
WIND_ENERGY_SCALE = 0.02 # k factor for E_wind = k * v_ij (adjusted to be in the [0.01, 0.05] range, pick a middle one)

# --- DiagnosticUAVEnv: Custom Gym Environment ---
class DiagnosticUAVEnv(gym.Env):
    """
    A custom Gym environment for a UAV navigating a grid to rescue victims
    while avoiding obstacles and managing resources (battery, time, risk).
    This version includes a redesigned action space, detailed energy model,
    wind profile, and refined reward function for a single UAV.
    """
    def __init__(self, debug=False):
        super(DiagnosticUAVEnv, self).__init__()
        self.debug = debug
        self.grid_size = (15, 15)
        self.grid_height, self.grid_width = self.grid_size

        # Observation space: 7 channels
        # (coverage, unrescued_victim_map, environmental_risk, obstacles, UAV_position, battery_status, mission_time)
        self.channels = 7
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.channels, *self.grid_size), dtype=np.float32)
        # Action space: 0: Explore Path, 1: Task Allocation, 2: Task Handover, 3: Reroute
        self.action_space = spaces.Discrete(4)
        self.max_steps = 150

        # Fixed locations for victims and obstacles for consistent environment
        self.initial_victims = [
            (0, 2), (1, 4), (2, 8), (3, 1), (4, 12), (5, 3), (6, 6), (7, 10),
            (8, 14), (9, 7), (10, 5), (11, 2), (12, 13), (13, 0), (14, 4)
        ]
        self.fixed_obstacles = [
            (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 7), (7, 8),
            (8, 9), (9, 10), (10, 11), (11, 12)
        ]

        self.total_initial_victims = len(self.initial_victims)
        self.base_risk_map = self._create_risk_map() # Base risk across the grid
        self.wind_map = self._create_wind_map() # Wind speed and direction for each cell

        self.action_names = ['Explore Path', 'Task Allocation', 'Task Handover', 'Reroute'] # For debugging output
        self.last_move_direction = None # To support 'Explore Path' action

    def _create_risk_map(self):
        """
        Creates a risk map for the environment. Areas near obstacles have higher risk.
        """
        risk_map = np.random.rand(self.grid_height, self.grid_width) * 0.2 + 0.1 # Base mild risk

        # Increase risk near fixed obstacles
        for (oy, ox) in self.fixed_obstacles:
            for y in range(max(0, oy - 1), min(self.grid_height, oy + 2)):
                for x in range(max(0, ox - 1), min(self.grid_width, ox + 2)):
                    distance = max(abs(y - oy), abs(x - ox))
                    if distance <= 1:
                        risk_map[y, x] += 0.2 * (1 - distance) # Risk decreases with distance from obstacle

        return np.clip(risk_map, 0.1, 0.5) # Clip risk values to a defined range

    def _create_wind_map(self):
        """
        Creates a wind map for the environment. Each cell has a wind speed and direction.
        Wind direction (theta_ij): 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
        Wind speed (v_ij): uniform random in [0, 1]
        """
        wind_map = np.empty(self.grid_size, dtype=object)
        for r in range(self.grid_height):
            for c in range(self.grid_width):
                speed = np.random.uniform(0, 1) # Wind speed v_ij in [0, 1]
                direction = np.random.randint(0, 4) # 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
                wind_map[r, c] = (speed, direction)
        return wind_map

    def reset(self):
        """
        Resets the environment to an initial state for a new episode.
        Initializes UAV position, maps, and resource levels.
        """
        self.step_count = 0
        self.energy_used = 0.0
        self.mission_time_elapsed = 0.0 # Renamed from mission_time
        self.cumulative_risk = 0.0 # Renamed from risk_score
        self.victims_rescued = 0

        # Initialize environment maps
        self.coverage_map = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        # victim_map now holds UNRESCUED victims. 1 for victim, 0 for no victim/rescued
        self.victim_map = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        self.obstacle_map = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)

        # Choose a random starting position for the UAV, not on an obstacle or victim
        valid_positions = [(y, x) for y in range(self.grid_height) for x in range(self.grid_width)
                           if (y, x) not in self.initial_victims and (y, x) not in self.fixed_obstacles]
        self.uav_pos = random.choice(valid_positions)
        self.last_move_direction = random.randint(0, 3) # Initialize a random starting direction

        # Populate static maps (victims are placed on initial_victims)
        for (y, x) in self.initial_victims:
            self.victim_map[y, x] = 1 # Mark all initial victims as present

        for (y, x) in self.fixed_obstacles:
            self.obstacle_map[y, x] = 1

        self.battery = 1.0 # Initial battery level (normalized, 0-1)
        self.time_left = 1.0 # Initial time remaining (normalized, 0-1)

        self.risk_map = self.base_risk_map.copy() # Current risk map

        # History to detect if the UAV is stuck
        self.position_history = deque(maxlen=5)
        self.position_history.append(self.uav_pos)

        # Reset reward components for debugging
        self.total_reward_components = {
            'victim_rescue': 0, 'exploration': 0, 'movement': 0,
            'battery_efficiency': 0, 'risk_avoidance': 0, 'stuck_penalty': 0,
            'time_efficiency': 0, 'completion_bonus': 0, 'no_move_penalty': 0,
            'step_reward': 0, 'action_cost_penalty': 0
        }

        # Calculate E_max_expected for energy efficiency reward
        avg_base_action_cost = (ACTION_ENERGY_COSTS[0] + ACTION_ENERGY_COSTS[1] +
                                ACTION_ENERGY_COSTS[2] + ACTION_ENERGY_COSTS[3]) / 4.0
        self.avg_E_step_estimate = BASE_ENERGY_STEP + avg_base_action_cost
        self.E_max_expected = self.max_steps * (self.avg_E_step_estimate * 1.5)

        return self.get_state()

    def _get_wind_energy_cost(self, position):
        """
        Calculates energy cost due to wind at a given position.
        E_wind = k * v_ij
        """
        # Temporarily return 0 for stability
        return 0 

    def _calculate_energy_cost(self, action, victim_rescued_this_step, moved_this_step, current_pos):
        """
        Calculates the total energy consumed for the current step based on the detailed model.
        ΔEⱼ(t) = E_base + E_action + E_task + E_env
        """
        E_base = BASE_ENERGY_STEP

        E_action = ACTION_ENERGY_COSTS.get(action, 0)

        E_task = TASK_ENERGY_RESCUE if victim_rescued_this_step else 0

        # Temporarily disable noise for stability
        E_noise = 0 
        E_wind = self._get_wind_energy_cost(current_pos)
        E_env = E_noise + E_wind

        total_energy_consumed = E_base + E_action + E_task + E_env
        return max(0, total_energy_consumed)

    def _get_unrescued_victim_positions(self):
        """Returns a list of (y, x) coordinates for unrescued victims."""
        return [(r, c) for r in range(self.grid_height) for c in range(self.grid_width) if self.victim_map[r, c] == 1]

    def _find_nearest_target(self, current_pos, target_coords_list):
        """
        Finds the nearest target from a list of coordinates (e.g., victims, unexplored cells).
        Returns (target_y, target_x) and its distance.
        """
        min_dist = float('inf')
        nearest_target = None
        current_y, current_x = current_pos

        for ty, tx in target_coords_list:
            if (ty, tx) in self.fixed_obstacles: continue

            dist = abs(current_y - ty) + abs(current_x - tx) # Manhattan distance
            if dist < min_dist:
                min_dist = dist
                nearest_target = (ty, tx)
        return nearest_target, min_dist

    def _get_next_step_towards(self, current_pos, target_pos):
        """
        Determines the next best step towards a target position using a greedy approach.
        """
        cy, cx = current_pos
        ty, tx = target_pos
        best_next_pos = current_pos
        min_dist_to_target = abs(cy - ty) + abs(cx - tx)

        possible_moves = [(cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)] # UP, DOWN, LEFT, RIGHT
        random.shuffle(possible_moves)

        for ny, nx in possible_moves:
            if 0 <= ny < self.grid_height and 0 <= nx < self.grid_width and (ny, nx) not in self.fixed_obstacles:
                dist = abs(ny - ty) + abs(nx - tx)
                if dist < min_dist_to_target:
                    min_dist_to_target = dist
                    best_next_pos = (ny, nx)
        return best_next_pos

    def _execute_action(self, action):
        """
        Executes the high-level action and determines UAV's movement.
        Returns new_uav_pos, moved_this_step, victim_rescued_this_step.
        """
        original_pos = self.uav_pos
        new_uav_pos = original_pos
        moved_this_step = False
        victim_rescued_this_step = False

        if action == 0: # Explore Path: Prioritize victims, then general exploration
            unrescued_victims = self._get_unrescued_victim_positions()
            target_pos = None

            if unrescued_victims:
                target_pos, _ = self._find_nearest_target(original_pos, unrescued_victims)
            
            if not target_pos: # No unrescued victims, or nearest_target couldn't find one, try to explore
                uncovered_cells = [(r, c) for r in range(self.grid_height) for c in range(self.grid_width)
                                   if self.coverage_map[r, c] == 0 and (r, c) not in self.fixed_obstacles]
                if uncovered_cells:
                    target_pos, _ = self._find_nearest_target(original_pos, uncovered_cells)

            if target_pos: # If a target (victim or unexplored) was found, move towards it
                new_uav_pos = self._get_next_step_towards(original_pos, target_pos)
            else: # Fallback: no victims, no unexplored, or cannot reach them, just take a random valid step
                new_uav_pos = self._take_random_valid_step(original_pos)
            
            if new_uav_pos != original_pos:
                moved_this_step = True

        elif action == 1: # Task Allocation: Conceptual action, no movement
            if self._get_unrescued_victim_positions():
                self.total_reward_components['action_cost_penalty'] += 0.05
            else:
                self.total_reward_components['action_cost_penalty'] += -0.1

        elif action == 2: # Task Handover: Conceptual action for single UAV, no movement, penalized
            self.total_reward_components['action_cost_penalty'] += -0.5

        elif action == 3: # Reroute: Take a random valid step
            new_uav_pos = self._take_random_valid_step(original_pos)
            if new_uav_pos != original_pos:
                moved_this_step = True

        # Check for victim rescue at new position AFTER potential movement
        if self.victim_map[new_uav_pos] == 1:
            self.victim_map[new_uav_pos] = 0
            self.victims_rescued += 1
            victim_rescued_this_step = True

        return new_uav_pos, moved_this_step, victim_rescued_this_step
    
    def _take_random_valid_step(self, current_pos):
        """Attempts to take a random step to a valid adjacent cell."""
        possible_moves = []
        cy, cx = current_pos
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]: # UP, DOWN, LEFT, RIGHT
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < self.grid_height and 0 <= nx < self.grid_width and (ny, nx) not in self.fixed_obstacles:
                possible_moves.append((ny, nx))
        
        if possible_moves:
            return random.choice(possible_moves)
        return current_pos


    def get_state(self):
        """
        Constructs the current observation (state) for the agent.
        Channel order:
        0: coverage_map
        1: victim_map (unrescued victims)
        2: risk_map
        3: obstacle_map
        4: uav_position_map
        5: battery_layer (broadcast)
        6: time_left_layer (broadcast)
        """
        uav_layer = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        uav_layer[self.uav_pos] = 1.0

        battery_layer = np.full((self.grid_height, self.grid_width), self.battery, dtype=np.float32)
        time_left_layer = np.full((self.grid_height, self.grid_width), self.time_left, dtype=np.float32)

        stacked = np.stack([
            self.coverage_map,
            self.victim_map,
            self.risk_map,
            self.obstacle_map,
            uav_layer,
            battery_layer,
            time_left_layer
        ], axis=0)

        return stacked

    def step(self, action):
        """
        Applies an action to the environment and calculates the next state, reward,
        and whether the episode is done.
        """
        self.step_count += 1
        prev_pos = self.uav_pos

        new_uav_pos, moved_this_step, victim_rescued_this_step = self._execute_action(action)
        self.uav_pos = new_uav_pos

        self.position_history.append(self.uav_pos)

        was_new_area = self.coverage_map[self.uav_pos] == 0
        self.coverage_map[self.uav_pos] = 1.0

        energy_consumed = self._calculate_energy_cost(action, victim_rescued_this_step, moved_this_step, self.uav_pos)
        self.energy_used += energy_consumed
        self.battery = max(0, self.battery - energy_consumed)

        time_step_cost = 1.0 / self.max_steps
        self.mission_time_elapsed += time_step_cost
        self.time_left = max(0, self.time_left - time_step_cost)

        self.cumulative_risk += self.risk_map[self.uav_pos] * 0.01

        done = (self.step_count >= self.max_steps or
                self.battery <= 0 or
                self.time_left <= 0 or
                self.victims_rescued == self.total_initial_victims)

        reward = self._calculate_detailed_reward(victim_rescued_this_step, prev_pos, moved_this_step, was_new_area, action)

        next_state = self.get_state()

        info = {
            'energy_used': self.energy_used,
            'mission_time_elapsed': self.mission_time_elapsed,
            'cumulative_risk': self.cumulative_risk,
            'battery': self.battery,
            'time_left': self.time_left,
            'step': self.step_count,
            'victims_rescued': self.victims_rescued,
            'total_victims': self.total_initial_victims,
            'victim_rescued_this_step': victim_rescued_this_step,
            'coverage_ratio': np.sum(self.coverage_map) / (self.grid_height * self.grid_width),
            'reward_components': self.total_reward_components.copy(),
            'action_taken': self.action_names[action],
            'position': self.uav_pos
        }

        return next_state, reward, done, info

    def _calculate_detailed_reward(self, victim_rescued, prev_pos, moved_this_step, was_new_area, action):
        """
        Calculates the reward for the current step based on various factors.
        --- IMPROVEMENT: Clamped total reward to a tighter range (-0.5 to 0.5). ---
        """
        reward_components = {
            'victim_rescue': 0, 'exploration': 0, 'movement': 0,
            'battery_efficiency': 0, 'risk_avoidance': 0, 'stuck_penalty': 0,
            'time_efficiency': 0, 'completion_bonus': 0, 'no_move_penalty': 0,
            'step_reward': 0, 'action_cost_penalty': 0
        }

        if victim_rescued:
            reward_components['victim_rescue'] = 50.0

        if was_new_area:
            reward_components['exploration'] = 5.0

        if action == 0 or action == 3:
            if moved_this_step:
                reward_components['movement'] = 0.1
                reward_components['no_move_penalty'] = 0
            else:
                reward_components['movement'] = -0.1
                reward_components['no_move_penalty'] = -0.75

        energy_efficiency_score = 1 - (self.energy_used / (self.E_max_expected + 1e-8))
        reward_components['battery_efficiency'] = energy_efficiency_score * 1.0

        max_possible_risk = self.max_steps * 0.5 * 0.01
        normalized_cumulative_risk = self.cumulative_risk / (max_possible_risk + 1e-8)
        risk_avoidance_score = 1 - normalized_cumulative_risk
        reward_components['risk_avoidance'] = risk_avoidance_score * 0.5

        if len(set(self.position_history)) <= 2 and self.step_count > 5:
            reward_components['stuck_penalty'] = -1.5

        reward_components['time_efficiency'] = self.time_left * 1.0

        reward_components['step_reward'] = 0.05

        if self.victims_rescued == self.total_initial_victims and self.victims_rescued > 0:
            efficiency_bonus = (self.battery + self.time_left) * 10
            reward_components['completion_bonus'] = 100.0 + efficiency_bonus

        for key, value in reward_components.items():
            self.total_reward_components[key] += value

        total_reward = sum(reward_components.values())
        # --- CRITICAL IMPROVEMENT: Clip the total reward per step to a very tight range ---
        total_reward = np.clip(total_reward, -0.5, 0.5) # Reduced from +/- 5.0 to +/- 0.5

        return total_reward

    def print_debug_info(self):
        """
        Prints detailed information about the episode's outcome,
        useful for debugging and understanding agent behavior.
        """
        print(f"Episode completed in {self.step_count} steps")
        print(f"Victims rescued: {self.victims_rescued}/{self.total_initial_victims}")
        print(f"Coverage: {np.sum(self.coverage_map)}/{self.grid_height * self.grid_width} "
              f"({np.sum(self.coverage_map)/(self.grid_height * self.grid_width)*100:.1f}%)")
        print(f"Final position: {self.uav_pos}")
        print(f"Battery remaining: {self.battery:.3f}")
        print(f"Time remaining: {self.time_left:.3f}")
        print("Total reward components breakdown:")
        for component, value in self.total_reward_components.items():
            if abs(value) > 0.01:
                print(f"    {component}: {value:.2f}")
        print("-" * 30)


# --- DiagnosticCNNActorCritic: Neural Network Model ---
class DiagnosticCNNActorCritic(nn.Module):
    """
    Convolutional Neural Network (CNN) based Actor-Critic model.
    """
    def __init__(self, input_channels=7, num_actions=4):
        super(DiagnosticCNNActorCritic, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(16 * 15 * 15, 128)
        self.fc2 = nn.Linear(128, 64)

        self.policy_head = nn.Linear(64, num_actions)
        self.value_head = nn.Linear(64, 1)

        self._init_weights()

    def _init_weights(self):
        """
        Initializes weights using orthogonal initialization, which can improve
        training stability for reinforcement learning models.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Defines the forward pass of the network.
        """
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        policy_logits = self.policy_head(x)
        value = self.value_head(x)

        return policy_logits, value


# --- DiagnosticPPOAgent: Proximal Policy Optimization Agent ---
class DiagnosticPPOAgent:
    """
    Implements the Proximal Policy Optimization (PPO) algorithm for training
    the DiagnosticCNNActorCritic model.
    """
    def __init__(self, model, lr=1e-6, gamma=0.99, eps_clip=0.2, k_epochs=4, entropy_coef=0.01):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef

        self.clear_memory()

    def clear_memory(self):
        """
        Clears the stored experiences after a policy update.
        """
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def select_action(self, state):
        """
        Given a state, uses the model to predict the best action, its log probability,
        and the state's value.
        """
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            logits, value = self.model(state_tensor)

            probs = torch.softmax(logits, dim=1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            return action.item(), log_prob.item(), value.item()

    def store_transition(self, state, action, reward, log_prob, value, done):
        """
        Stores a single step's experience (transition) into the agent's memory.
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def update_with_loss_tracking(self):
        """
        Performs the PPO policy update using the collected experiences.
        --- IMPROVEMENT: Implemented mini-batching for stability. ---
        """
        if len(self.states) == 0:
            return 0.0, 0.0, 0.0, 0.0

        states = torch.tensor(np.array(self.states), dtype=torch.float32).to(device)
        actions = torch.tensor(self.actions, dtype=torch.long).to(device)
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32).to(device)
        old_values = torch.tensor(self.values, dtype=torch.float32).to(device)

        returns = self._calculate_returns()
        advantages = returns - old_values

        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Prepare for mini-batching
        dataset_size = len(self.states)
        indices = np.arange(dataset_size)
        mini_batch_size = 64 # Or adjust based on your collected data size

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        num_batches = 0

        for _ in range(self.k_epochs):
            np.random.shuffle(indices)
            for start_idx in range(0, dataset_size, mini_batch_size):
                end_idx = min(start_idx + mini_batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                logits, values = self.model(batch_states)
                probs = torch.softmax(logits, dim=1)
                dist = torch.distributions.Categorical(probs)

                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy()

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.MSELoss()(values.squeeze(), batch_returns)

                entropy_loss = -self.entropy_coef * entropy.mean()

                loss = policy_loss + 1.0 * value_loss + entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                # --- IMPROVEMENT: Gradient clipping reduced to 0.5 ---
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                total_actor_loss += policy_loss.item()
                total_critic_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_batches += 1

        self.clear_memory()
        
        if num_batches == 0:
            return 0.0, 0.0, 0.0, 0.0
        
        avg_loss = (total_actor_loss + total_critic_loss) / num_batches
        return (avg_loss,
                total_actor_loss / num_batches,
                total_critic_loss / num_batches,
                total_entropy / num_batches)

    def _calculate_returns(self):
        """
        Calculates discounted cumulative rewards (returns) for each step
        in the collected trajectory.
        """
        returns = []
        discounted_sum = 0

        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                discounted_sum = 0
            discounted_sum = reward + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)

        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        return returns


# --- Training Function ---
def detailed_train_with_loss_tracking():
    """
    Main training function. Sets up the environment, model, and agent,
    then runs the training loop, logs metrics, and saves results.
    """
    env = DiagnosticUAVEnv(debug=True)
    model = DiagnosticCNNActorCritic(input_channels=env.channels).to(device)
    agent = DiagnosticPPOAgent(model)

    num_episodes = 3000
    batch_size = 16
    print_interval = 50

    episode_rewards = []
    episode_victims = []
    episode_coverage = []
    episode_steps = []
    training_losses = []
    actor_losses = []
    critic_losses = []
    entropy_values = []

    print("Starting diagnostic training with loss tracking...")

    episode_batch = 0
    best_victims = 0

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        steps = 0

        while True:
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            agent.store_transition(state, action, reward, log_prob, value, done)

            episode_reward += reward
            steps += 1
            state = next_state

            if done:
                break

        episode_rewards.append(episode_reward)
        episode_victims.append(info['victims_rescued'])
        episode_coverage.append(info['coverage_ratio'])
        episode_steps.append(steps)
        episode_batch += 1

        if episode_batch >= batch_size:
            avg_loss, actor_loss, critic_loss, entropy = agent.update_with_loss_tracking()
            training_losses.append(avg_loss)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            entropy_values.append(entropy)
            episode_batch = 0

        if (episode + 1) % print_interval == 0:
            recent_episodes = min(print_interval, len(episode_rewards))
            avg_reward = np.mean(episode_rewards[-recent_episodes:])
            avg_victims = np.mean(episode_victims[-recent_episodes:])

            print(f"Episode {episode+1}/{num_episodes}")
            print(f"    Avg Reward (last {recent_episodes}): {avg_reward:.2f}")
            print(f"    Avg Victims (last {recent_episodes}): {avg_victims:.2f}/{env.total_initial_victims}")
            if training_losses:
                print(f"    Recent Losses - Total: {training_losses[-1]:.4f}, Actor: {actor_losses[-1]:.4f}, Critic: {critic_losses[-1]:.4f}, Entropy: {entropy_values[-1]:.4f}")
            print()

    # --- Save Results ---
    os.makedirs("model_output", exist_ok=True)
    torch.save(model.state_dict(), "model_output/uav_diagnostic_model_v3.pth")
    print(f"Model saved to model_output/uav_diagnostic_model_v2.pth")

    results_df = pd.DataFrame({
        'episode': range(len(episode_rewards)),
        'reward': episode_rewards,
        'victims_rescued': episode_victims,
        'coverage_ratio': episode_coverage,
        'steps': episode_steps,
        'total_loss': training_losses + [np.nan] * (len(episode_rewards) - len(training_losses)),
        'actor_loss': actor_losses + [np.nan] * (len(episode_rewards) - len(actor_losses)),
        'critic_loss': critic_losses + [np.nan] * (len(episode_rewards) - len(critic_losses)),
        'entropy': entropy_values + [np.nan] * (len(episode_rewards) - len(entropy_values))
    })
    results_df.to_csv("model_output/uav_training_results_v2.csv", index=False)
    print(f"Training results saved to model_output/uav_training_results_v3.csv")

    return results_df

# --- Plotting Function ---
def plot_training_results(results_path="model_output/uav_training_results_v3.csv",
                          start_episode=0, end_episode=None):
    """
    Loads training results from a CSV and generates plots for reward, loss components,
    and performance metrics.
    """
    if not os.path.exists(results_path):
        print(f"Error: Results file not found at {results_path}. Please run training first.")
        return

    df = pd.read_csv(results_path)

    env = DiagnosticUAVEnv(debug=False)

    if end_episode is None:
        end_episode = len(df)
    df = df[(df['episode'] >= start_episode) & (df['episode'] < end_episode)]

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(12, 25))

    ax1.plot(df['episode'], df['reward'], label='Episode Reward', color='blue', alpha=0.6)
    ax1.set_title(f'Training Rewards (Episodes {start_episode}-{end_episode})')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(True)
    ax1.legend()

    ax2.plot(df['episode'], df['total_loss'], label='Total Loss', color='red')
    ax2.set_title(f'Total Loss (Episodes {start_episode}-{end_episode})')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    ax3.plot(df['episode'], df['critic_loss'], label='Critic Loss', color='purple')
    ax3.set_title(f'Critic Loss (Episodes {start_episode}-{end_episode})')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.grid(True)

    ax4.plot(df['episode'], df['actor_loss'], label='Actor Loss', color='green')
    ax4.set_title(f'Actor Loss (Episodes {start_episode}-{end_episode})')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Loss')
    ax4.legend()
    ax4.grid(True)

    ax5.plot(df['episode'], df['victims_rescued'], label='Victims Rescued', color='orange')
    ax5.plot(df['episode'], df['coverage_ratio'] * (env.grid_height * env.grid_width),
             label=f'Coverage (scaled to {env.grid_height * env.grid_width} cells)', color='cyan', alpha=0.5)
    ax5.set_title(f'Performance Metrics (Episodes {start_episode}-{end_episode})')
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Count / Ratio')
    ax5.legend()
    ax5.grid(True)

    plt.tight_layout()
    os.makedirs("model_output/plots", exist_ok=True)
    plot_filename = f"model_output/plots/training_metrics_{start_episode}_to_{end_episode}3.png"
    plt.savefig(plot_filename)
    print(f"Plots saved to {plot_filename}")
    plt.show()


# --- Simulation Function (for checking trained model behavior) ---
def run_simulation_and_get_trajectory(model_path="model_output/uav_diagnostic_model_v3.pth"):
    """
    Runs a single simulation episode using the trained model and collects the trajectory data.
    """
    env = DiagnosticUAVEnv(debug=True)
    model = DiagnosticCNNActorCritic(input_channels=env.channels).to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Loaded model from {model_path} for simulation.")
    else:
        print(f"Warning: Model not found at {model_path}. Running simulation with an untrained model.")
        model.eval()

    agent = DiagnosticPPOAgent(model)

    trajectory = []
    state = env.reset()
    done = False
    
    initial_step_data = {
        'uav_pos': env.uav_pos,
        'victim_map': env.victim_map.tolist(),
        'obstacle_map': env.obstacle_map.tolist(),
        'grid_height': env.grid_height,
        'grid_width': env.grid_width,
        'step_count': env.step_count,
        'victims_rescued': env.victims_rescued,
        'total_victims': env.total_initial_victims
    }
    trajectory.append(initial_step_data)

    print("Running simulation episode...")
    while not done:
        action, _, _ = agent.select_action(state)
        next_state, reward, done, info = env.step(action)

        current_step_data = {
            'uav_pos': info['position'],
            'victim_map': env.victim_map.tolist(),
            'obstacle_map': env.obstacle_map.tolist(),
            'grid_height': env.grid_height,
            'grid_width': env.grid_width,
            'step_count': info['step'],
            'victims_rescued': info['victims_rescued'],
            'total_victims': info['total_initial_victims']
        }
        trajectory.append(current_step_data)

        state = next_state

        if info['step'] >= env.max_steps and not done:
            done = True
            print("Simulation terminated early: Reached max_steps.")

    print(f"Simulation completed. Total steps: {len(trajectory)-1}.")
    env.print_debug_info()

    return trajectory


# --- Main Execution Block ---
if __name__ == "__main__":
    os.makedirs("model_output", exist_ok=True)

    results_file = "model_output/uav_training_results_v3.csv"
    model_file = "model_output/uav_diagnostic_model_v3.pth"

    if not os.path.exists(results_file) or not os.path.exists(model_file):
        print("Training results or model not found. Starting training...")
        detailed_train_with_loss_tracking()
    else:
        print("Training results and model found. Skipping training.")

    print("\nPlotting training results...")
    plot_training_results(end_episode=3000)

    print("\nRunning a simulation episode to inspect behavior...")
    simulation_trajectory = run_simulation_and_get_trajectory()

    print(f"\nSimulation trajectory collected. It contains {len(simulation_trajectory)} steps.")
    print("Example of first step in trajectory:", simulation_trajectory[0])