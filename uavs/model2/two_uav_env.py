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

# Set device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class DiagnosticUAVEnv(gym.Env):
    def __init__(self, debug=False):
        super(DiagnosticUAVEnv, self).__init__()
        self.debug = debug
        self.grid_size = (15, 15)
        self.grid_height, self.grid_width = self.grid_size

        # Channels: coverage, victim, rescued, obstacle, UAV1, UAV2, battery1, battery2, time_left, risk_map
        self.channels = 10 
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.channels, *self.grid_size), dtype=np.float32)
        
        # Action space for two UAVs: 4 actions for UAV1, 4 for UAV2 => 4*4 = 16 discrete actions
        # Action mapping: action = action1 * 4 + action2
        self.action_space = spaces.Discrete(16) 

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
        self.total_energy_consumption = 0.0 # To track total energy for plotting
        self.total_risk_score = 0.0 # To track total risk for plotting

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
        self.energy_used1 = 0.0
        self.energy_used2 = 0.0
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
        self.uav_pos1 = random.choice(valid_positions)
        valid_positions.remove(self.uav_pos1)
        self.uav_pos2 = random.choice(valid_positions)

        for (y, x) in self.fixed_victims:
            self.victim_map[y, x] = 1
        for (y, x) in self.fixed_obstacles:
            self.obstacle_map[y, x] = 1

        self.battery1 = 1.0
        self.battery2 = 1.0
        self.time_left = 1.0
        self.risk_map = self.base_risk_map.copy()

        self.position_history1 = deque(maxlen=10)
        self.position_history2 = deque(maxlen=10)
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

        # Reset total energy/risk for plotting
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
        prev_pos1 = self.uav_pos1
        prev_pos2 = self.uav_pos2

        # Decode combined action
        action1 = action // 4
        action2 = action % 4

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

        # --- Collision Avoidance Logic ---
        # Did UAV1 attempt to move into an obstacle?
        uav1_collided_obstacle = (potential_pos1 in self.fixed_obstacles)

        # Did UAV2 attempt to move into an obstacle?
        uav2_collided_obstacle = (potential_pos2 in self.fixed_obstacles)
        
        # Scenario 1: Both UAVs try to move to the same *new* cell (head-on collision)
        # This implies potential_pos1 == potential_pos2 and neither is their current position
        if potential_pos1 == potential_pos2 and potential_pos1 != self.uav_pos1 and potential_pos1 != self.uav_pos2:
            # Both UAVs stay put
            actual_pos1 = self.uav_pos1
            actual_pos2 = self.uav_pos2
        # Scenario 2: UAV1 moves to UAV2's current position, and UAV2 stays put or moves elsewhere
        elif potential_pos1 == self.uav_pos2 and potential_pos2 != self.uav_pos1:
            actual_pos1 = self.uav_pos1 # UAV1 stays put
            actual_pos2 = potential_pos2 if not uav2_collided_obstacle and potential_pos2 != self.uav_pos1 else self.uav_pos2
        # Scenario 3: UAV2 moves to UAV1's current position, and UAV1 stays put or moves elsewhere
        elif potential_pos2 == self.uav_pos1 and potential_pos1 != self.uav_pos2:
            actual_pos2 = self.uav_pos2 # UAV2 stays put
            actual_pos1 = potential_pos1 if not uav1_collided_obstacle and potential_pos1 != self.uav_pos2 else self.uav_pos1
        # Scenario 4: UAVs try to swap positions
        elif potential_pos1 == self.uav_pos2 and potential_pos2 == self.uav_pos1:
            # Both UAVs stay put to avoid swapping collision
            actual_pos1 = self.uav_pos1
            actual_pos2 = self.uav_pos2
        # Scenario 5: No collision, or only one UAV attempts to move into the other's current spot
        else:
            # Check for current position overlap *after* potential move
            # UAV1 can move to potential_pos1 if it's not an obstacle and not potential_pos2
            if not uav1_collided_obstacle and potential_pos1 != potential_pos2:
                actual_pos1 = potential_pos1
            else:
                actual_pos1 = self.uav_pos1 # Stay put if obstacle or targetting other UAV's new spot

            # UAV2 can move to potential_pos2 if it's not an obstacle and not actual_pos1 (which might be UAV1's new spot)
            if not uav2_collided_obstacle and potential_pos2 != actual_pos1:
                actual_pos2 = potential_pos2
            else:
                actual_pos2 = self.uav_pos2 # Stay put if obstacle or targetting other UAV's new spot
        
        self.uav_pos1 = actual_pos1
        self.uav_pos2 = actual_pos2
        # --- End Collision Avoidance Logic ---

        self.position_history1.append(self.uav_pos1)
        self.position_history2.append(self.uav_pos2)

        # Update coverage map (both UAVs contribute)
        was_new_area1 = self.coverage_map[self.uav_pos1] == 0
        was_new_area2 = self.coverage_map[self.uav_pos2] == 0
        was_new_area = was_new_area1 or was_new_area2
        self.coverage_map[self.uav_pos1] = 1.0
        self.coverage_map[self.uav_pos2] = 1.0

        # Check for victim rescue (if either UAV rescues)
        victim_rescued = False
        if self.victim_map[self.uav_pos1] == 1 and self.rescued_map[self.uav_pos1] == 0:
            self.rescued_map[self.uav_pos1] = 1
            self.victims_rescued += 1
            victim_rescued = True
        if self.victim_map[self.uav_pos2] == 1 and self.rescued_map[self.uav_pos2] == 0:
            self.rescued_map[self.uav_pos2] = 1
            self.victims_rescued += 1
            victim_rescued = True

        # Simplified energy model for each UAV
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

        self.energy_used1 += energy_consumed1
        self.battery1 = max(0, self.battery1 - energy_consumed1)
        self.energy_used2 += energy_consumed2
        self.battery2 = max(0, self.battery2 - energy_consumed2)
        
        # Accumulate total energy for plotting
        self.total_energy_consumption += (energy_consumed1 + energy_consumed2)

        time_step = 1.0 / self.max_steps
        self.mission_time += time_step
        self.time_left = max(0, self.time_left - time_step)

        # Risk accumulation from both UAVs
        self.risk_score += (self.risk_map[self.uav_pos1] * 0.01 + self.risk_map[self.uav_pos2] * 0.01)
        self.total_risk_score += (self.risk_map[self.uav_pos1] * 0.01 + self.risk_map[self.uav_pos2] * 0.01)


        # Check termination
        done = (self.step_count >= self.max_steps or
                (self.battery1 <= 0 and self.battery2 <= 0) or # Both batteries dead
                self.time_left <= 0 or
                self.victims_rescued == self.total_victims)

        reward = self._calculate_detailed_reward(victim_rescued, prev_pos1, prev_pos2, was_new_area)

        next_state = self.get_state()

        info = {
            'energy_used1': self.energy_used1,
            'energy_used2': self.energy_used2,
            'mission_time': self.mission_time,
            'risk_score': self.risk_score,
            'battery1': self.battery1,
            'battery2': self.battery2,
            'time_left': self.time_left,
            'step': self.step_count,
            'victims_rescued': self.victims_rescued,
            'total_victims': self.total_victims,
            'victim_rescued_this_step': victim_rescued,
            'coverage_ratio': np.sum(self.coverage_map) / (self.grid_height * self.grid_width),
            'reward_components': self.total_reward_components.copy(),
            'action_taken_uav1': self.action_names[action1],
            'action_taken_uav2': self.action_names[action2],
            'position_uav1': self.uav_pos1,
            'position_uav2': self.uav_pos2,
            'total_energy_consumption': self.total_energy_consumption, # For plotting
            'cumulative_risk_score': self.total_risk_score # For plotting
        }

        return next_state, reward, done, info

    def _calculate_detailed_reward(self, victim_rescued, prev_pos1, prev_pos2, was_new_area):
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
        
        # Penalty for staying in the same spot due to collision, not chosen by agent
        # If the agent tried to move (action1/2 indicates movement) but the UAV didn't move because of collision, penalize
        # This is implicitly covered by the existing 'movement' penalty (-0.2 for not moving).
        # We don't need an *additional* explicit collision penalty here, as not moving is already penalized.

        # Combined energy penalty
        reward_components['energy_penalty'] = -(self.energy_used1 + self.energy_used2) * 0.4

        # Combined risk penalty
        reward_components['risk_penalty'] = -(self.risk_map[self.uav_pos1] * 0.15 + self.risk_map[self.uav_pos2] * 0.15)

        # Stuck penalty if both UAVs are stuck
        are_uav1_stuck = len(set(self.position_history1)) <= 3
        are_uav2_stuck = len(set(self.position_history2)) <= 3
        if are_uav1_stuck and are_uav2_stuck and self.step_count > self.max_steps * 0.1:
            reward_components['stuck_penalty'] = -2.0

        reward_components['time_penalty'] = -0.02

        if self.victims_rescued == self.total_victims:
            efficiency_bonus = ((self.battery1 + self.battery2) / 2 * 20.0 + self.time_left * 20.0) # Average battery
            reward_components['completion_bonus'] = 100.0 + efficiency_bonus

        for key, value in reward_components.items():
            self.total_reward_components[key] += value

        total_reward = sum(reward_components.values())
        return total_reward

    def print_debug_info(self):
        print(f"Episode completed in {self.step_count} steps")
        print(f"Victims rescued: {self.victims_rescued}/{self.total_victims}")
        print(f"Coverage: {np.sum(self.coverage_map)}/{self.grid_height * self.grid_width} ({np.sum(self.coverage_map)/(self.grid_height * self.grid_width)*100:.1f}%)")
        print(f"UAV1 final position: {self.uav_pos1}")
        print(f"UAV2 final position: {self.uav_pos2}")
        print(f"Battery1 remaining: {self.battery1:.3f}")
        print(f"Battery2 remaining: {self.battery2:.3f}")
        print(f"Time remaining: {self.time_left:.3f}")
        print("Reward components:")
        for component, value in self.total_reward_components.items():
            if abs(value) > 0.01:
                print(f"   {component}: {value:.2f}")
        print()

class DiagnosticCNNActorCritic(nn.Module):
    def __init__(self, input_channels=10, num_actions=16): # Updated input_channels and num_actions
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

class DiagnosticPPOAgent:
    def __init__(self, model, lr=1e-3, gamma=0.99, eps_clip=0.3, k_epochs=8, entropy_coef=0.005):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef

        self.clear_memory()

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            logits, value = self.model(state_tensor)

            probs = torch.softmax(logits, dim=1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            return action.item(), log_prob.item(), value.item()

    def store_transition(self, state, action, reward, log_prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def update_with_loss_tracking(self):
        if len(self.states) == 0:
            print("Memory is empty, no update.")
            return 0.0, 0.0, 0.0, 0.0
        
        states = torch.tensor(np.array(self.states), dtype=torch.float32).to(device)
        actions = torch.tensor(self.actions, dtype=torch.long).to(device)
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32).to(device)
        old_values = torch.tensor(self.values, dtype=torch.float32).to(device)
        
        returns = self._calculate_returns()
        advantages = returns - old_values
        
        # Normalize advantages for more stable training
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = torch.zeros_like(advantages)
        
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        
        for epoch in range(self.k_epochs):
            logits, values = self.model(states)
            probs = torch.softmax(logits, dim=1)
            dist = torch.distributions.Categorical(probs)
            
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            
            # PPO clipping
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = nn.MSELoss()(values.squeeze(), returns)
            
            entropy_loss = -self.entropy_coef * entropy.mean()
            
            loss = policy_loss + 0.5 * value_loss + entropy_loss
            
            total_actor_loss += policy_loss.item()
            total_critic_loss += value_loss.item()
            total_entropy += entropy.mean().item()
            
            self.optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
        
        avg_actor_loss = total_actor_loss / self.k_epochs
        avg_critic_loss = total_critic_loss / self.k_epochs
        avg_entropy = total_entropy / self.k_epochs

        self.clear_memory() # Clear memory after update
        return (avg_actor_loss + 0.5 * avg_critic_loss + avg_entropy), avg_actor_loss, avg_critic_loss, avg_entropy

    def _calculate_returns(self):
        returns = []
        discounted_sum = 0

        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                discounted_sum = 0
            discounted_sum = reward + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)

        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        return returns

def detailed_train_with_loss_tracking():
    env = DiagnosticUAVEnv(debug=False)
    model = DiagnosticCNNActorCritic(input_channels=env.channels, num_actions=env.action_space.n).to(device)

    agent = DiagnosticPPOAgent(model, lr=1e-3, eps_clip=0.3, k_epochs=8, entropy_coef=0.005)

    num_episodes = 1000
    batch_size = 32
    print_interval = 100

    episode_rewards = []
    episode_victims = []
    episode_coverage = []
    episode_steps = []
    episode_avg_battery = []
    episode_total_energy_consumed = []
    episode_cumulative_risk = []

    training_losses = []
    actor_losses = []
    critic_losses = []
    entropy_values = []

    print("Starting diagnostic training with **TWO UAVs (No Overlap)** and improved parameters...")

    episode_batch_counter = 0
    best_victims = 0

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        steps = 0
        current_episode_energy_consumed = 0.0
        current_episode_risk = 0.0

        while True:
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            agent.store_transition(state, action, reward, log_prob, value, done)

            episode_reward += reward
            steps += 1
            state = next_state
            
            current_episode_energy_consumed = info['total_energy_consumption']
            current_episode_risk = info['cumulative_risk_score']

            if done:
                break

        episode_rewards.append(episode_reward)
        episode_victims.append(info['victims_rescued'])
        episode_coverage.append(info['coverage_ratio'])
        episode_steps.append(steps)
        episode_avg_battery.append((info['battery1'] + info['battery2']) / 2)
        episode_total_energy_consumed.append(current_episode_energy_consumed)
        episode_cumulative_risk.append(current_episode_risk)

        episode_batch_counter += 1

        if episode_batch_counter >= batch_size:
            avg_loss, actor_loss, critic_loss, entropy = agent.update_with_loss_tracking()
            training_losses.append(avg_loss)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            entropy_values.append(entropy)
            episode_batch_counter = 0

        if (episode + 1) % print_interval == 0:
            recent_episodes = min(print_interval, len(episode_rewards))
            avg_reward = np.mean(episode_rewards[-recent_episodes:])
            avg_victims = np.mean(episode_victims[-recent_episodes:])
            avg_coverage = np.mean(episode_coverage[-recent_episodes:])
            avg_battery = np.mean(episode_avg_battery[-recent_episodes:])
            avg_energy_consumed = np.mean(episode_total_energy_consumed[-recent_episodes:])
            avg_cumulative_risk = np.mean(episode_cumulative_risk[-recent_episodes:])


            print(f"Episode {episode+1}/{num_episodes}")
            print(f"   Avg Reward (last {recent_episodes}): {avg_reward:.2f}")
            print(f"   Avg Victims (last {recent_episodes}): {avg_victims:.2f}/{env.total_victims}")
            print(f"   Avg Coverage Ratio (last {recent_episodes}): {avg_coverage:.3f}")
            print(f"   Avg Final Battery (last {recent_episodes}): {avg_battery:.3f}")
            print(f"   Avg Total Energy Consumed (last {recent_episodes}): {avg_energy_consumed:.3f}")
            print(f"   Avg Cumulative Risk (last {recent_episodes}): {avg_cumulative_risk:.3f}")


            if training_losses and actor_losses and critic_losses:
                print(f"   Recent Losses - Total: {training_losses[-1]:.4f}, Actor: {actor_losses[-1]:.4f}, Critic: {critic_losses[-1]:.4f}, Entropy: {entropy_values[-1]:.4f}")
            print("-" * 40)

        if (episode + 1) % (print_interval * 2) == 0:
            current_avg_victims = np.mean(episode_victims[max(0, episode-print_interval*2):episode+1])
            if current_avg_victims > best_victims:
                best_victims = current_avg_victims
                torch.save(model.state_dict(), "model_output/two_uavs.pth") # New model name
                print(f"--- Saved BEST model at Episode {episode+1} with Avg Victims: {best_victims:.2f} ---")

    os.makedirs("model_output", exist_ok=True)
    torch.save(model.state_dict(), "model_output/two_uavs.pth") # New model name

    results_df = pd.DataFrame({
        'episode': range(len(episode_rewards)),
        'reward': episode_rewards,
        'victims_rescued': episode_victims,
        'coverage_ratio': episode_coverage,
        'steps': episode_steps,
        'avg_final_battery': episode_avg_battery,
        'total_energy_consumed': episode_total_energy_consumed,
        'cumulative_risk': episode_cumulative_risk
    })
    loss_episode_indices = [i * batch_size for i in range(len(training_losses))]
    loss_df = pd.DataFrame({
        'episode': loss_episode_indices,
        'total_loss': training_losses,
        'actor_loss': actor_losses,
        'critic_loss': critic_losses,
        'entropy': entropy_values
    })
    results_df = pd.merge(results_df, loss_df, on='episode', how='left')
    results_df.to_csv("model_output/two_uavs.csv", index=False) # New results filename

    return results_df

def plot_training_results(results_path="model_output/two_uavs.csv", # New path
                          start_episode=0, end_episode=None):
    df = pd.read_csv(results_path)
    env = DiagnosticUAVEnv(debug=False)

    if end_episode is None:
        end_episode = len(df)
    df = df[(df['episode'] >= start_episode) & (df['episode'] <= end_episode)]

    fig, axes = plt.subplots(5, 1, figsize=(12, 25))

    # Plot 1: Reward
    axes[0].plot(df['episode'], df['reward'], label='Episode Reward', color='blue', alpha=0.6)
    axes[0].set_title(f'Training Rewards (Episodes {start_episode}-{end_episode})')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].grid(True)
    axes[0].legend()

    # Plot 2: Losses
    df_losses = df.dropna(subset=['actor_loss', 'critic_loss'])
    axes[1].plot(df_losses['episode'], df_losses['actor_loss'], label='Actor Loss', color='green')
    axes[1].plot(df_losses['episode'], df_losses['critic_loss'], label='Critic Loss', color='purple')
    axes[1].set_title(f'Training Losses (Episodes {start_episode}-{end_episode})')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    # Plot 3: Performance Metrics (Victims & Coverage)
    axes[2].plot(df['episode'], df['victims_rescued'], label='Victims Rescued', color='orange')
    axes[2].plot(df['episode'], df['coverage_ratio'] * env.grid_height * env.grid_width,
                 label='Coverage (cells visited)', color='cyan', alpha=0.5)
    axes[2].set_title(f'Performance Metrics (Episodes {start_episode}-{end_episode})')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Count')
    axes[2].legend()
    axes[2].grid(True)

    # Plot 4: Average Final Battery
    axes[3].plot(df['episode'], df['avg_final_battery'], label='Avg Final Battery (UAV1+UAV2)/2', color='red')
    axes[3].set_title(f'Average Final Battery (Episodes {start_episode}-{end_episode})')
    axes[3].set_xlabel('Episode')
    axes[3].set_ylabel('Battery Level (0-1)')
    axes[3].set_ylim(0, 1)
    axes[3].legend()
    axes[3].grid(True)

    # Plot 5: Cumulative Risk and Total Energy Consumed
    axes[4].plot(df['episode'], df['cumulative_risk'], label='Cumulative Risk', color='brown')
    axes[4].plot(df['episode'], df['total_energy_consumed'], label='Total Energy Consumed', color='blueviolet', linestyle='--')
    axes[4].set_title(f'Cumulative Risk and Total Energy Consumed (Episodes {start_episode}-{end_episode})')
    axes[4].set_xlabel('Episode')
    axes[4].set_ylabel('Value')
    axes[4].legend()
    axes[4].grid(True)

    plt.tight_layout()
    plot_filename = f"model_output/plots/two_uavs_{start_episode}_to_{end_episode}.png" # New plot filename
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    plt.savefig(plot_filename)
    plt.show()

if __name__ == "__main__":
    results_file_two_uavs_no_overlap = "model_output/two_uavs.csv"

    if not os.path.exists(results_file_two_uavs_no_overlap):
        print("Running training for TWO UAVs with NO OVERLAP constraint...")
        detailed_train_with_loss_tracking()
    else:
        print(f"Found existing results for two UAVs (no overlap) at {results_file_two_uavs_no_overlap}. Skipping training.")

    print("Plotting results for two UAVs (no overlap)...")
    plot_training_results(results_path=results_file_two_uavs_no_overlap, end_episode=1000)