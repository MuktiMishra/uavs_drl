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

        # Initialize coverage map, victim map, obstacle map
        self.coverage_map = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        self.victim_map = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        self.obstacle_map = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)

        # Place victims and obstacles
        for (y, x) in self.fixed_victims:
            self.victim_map[y, x] = 1
        for (y, x) in self.fixed_obstacles:
            self.obstacle_map[y, x] = 1

        # Initialize UAV position (ensure it doesn't start on an obstacle or victim)
        self.uav_pos = (
            np.random.randint(self.grid_height),
            np.random.randint(self.grid_width)
        )
        while self.uav_pos in self.fixed_victims or self.uav_pos in self.fixed_obstacles:
            self.uav_pos = (
                np.random.randint(self.grid_height),
                np.random.randint(self.grid_width)
            )
        
        # Mark the initial UAV position as covered
        self.coverage_map[self.uav_pos] = 1.0

        self.battery = 1.0 # Represents 100% battery
        self.time_left = 1.0 # Represents 100% time left
        self.risk_map = np.random.rand(self.grid_height, self.grid_width) # Random risk values for each cell

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

    def step(self, action):
        self.step_count += 1
        prev_pos = self.uav_pos
        current_y, current_x = self.uav_pos # Store current position before potential move

        y, x = current_y, current_x # Start with current UAV position
        # Determine potential new position based on action
        if action == 0: # Up
            y = max(0, y - 1)
        elif action == 1: # Down
            y = min(self.grid_height - 1, y + 1)
        elif action == 2: # Left
            x = max(0, x - 1)
        elif action == 3: # Right
            x = min(self.grid_width - 1, x + 1)

        new_uav_pos = (y, x) # Potential new position

        step_reward = 0.0 # Initialize reward for this specific step

        # --- Handle Obstacles ---
        if new_uav_pos in self.fixed_obstacles:
            # If moving into an obstacle, stay at prev_pos and apply a significant penalty
            self.uav_pos = prev_pos # UAV does not move
            step_reward -= 2.0 # Severe penalty for hitting an obstacle
        else:
            self.uav_pos = new_uav_pos # Update UAV position if no obstacle

            # --- Reward for exploring new cell ---
            if self.coverage_map[self.uav_pos] == 0:
                step_reward += 0.1 # Small positive reward for new discovery (tune this)
                self.coverage_map[self.uav_pos] = 1.0 # Mark as covered
            else:
                # --- Penalty for revisiting an already explored cell (if not stuck) ---
                # This only applies if the UAV actually moved to an already visited cell
                if self.uav_pos != prev_pos:
                    step_reward -= 0.02 # Small penalty for revisiting (tune this)

            # --- High reward for reaching a victim ---
            if self.victim_map[self.uav_pos] == 1:
                step_reward += 5.0 # Very high reward for finding a victim (tune this)
                self.victim_map[self.uav_pos] = 0 # Mark victim as found (so it's not rewarded again)

        # --- Calculate Energy Consumption ---
        E_base = np.random.uniform(0.002, 0.01) # Base energy consumption per step
        E_action = np.random.uniform(0.001, 0.015) # Energy for performing an action
        E_task = np.random.uniform(0.001, 0.02) # Energy for performing a task (e.g., scanning)
        E_env = np.random.normal(0.0012, 0.0001) # Environmental energy factors (e.g., wind)
        energy_consumed_this_step = E_base + E_action + E_task + E_env
        self.energy_used += energy_consumed_this_step
        self.battery -= energy_consumed_this_step # Battery decreases

        # --- Calculate Mission Time ---
        time_taken_this_step = np.random.uniform(0.1, 1.0)
        self.mission_time += time_taken_this_step
        self.time_left -= time_taken_this_step / self.max_steps # Normalize time left

        # --- Calculate Risk Score ---
        risk_at_current_pos = self.risk_map[self.uav_pos]
        self.risk_score += risk_at_current_pos

        # --- Determine Episode Termination ---
        done = self.step_count >= self.max_steps or self.battery <= 0 or self.time_left <= 0

        # --- Calculate F-score components (adjusted for realistic max values) ---
        # Estimated maximum possible accumulated energy over max_steps
        # Using typical average values from your random distributions
        avg_energy_per_step = (0.002 + 0.01 + 0.001 + 0.015 + 0.001 + 0.02 + 0.0012) / 7
        E_max_expected = self.max_steps * (avg_energy_per_step * 1.5) # Adding a buffer for variability

        # Maximum possible accumulated mission time over max_steps
        tau_max_expected = self.max_steps * 1.0 # If each step takes 1.0 unit of time

        # Maximum possible accumulated risk score (if always in a 1.0 risk cell)
        R_max_expected = self.max_steps * 1.0

        # F-score components (clamped between 0 and 1, or can go negative for over-budget)
        f2 = 1 - (self.energy_used / E_max_expected)
        f3 = 1 - (self.mission_time / tau_max_expected)
        f4 = 1 - (self.risk_score / R_max_expected)

        # --- Combine all reward components ---
        base_mission_reward = (f2 + f3 + f4) / 3 # Averaged contribution from efficiency and safety

        # Total reward for this step
        reward = base_mission_reward + step_reward

        # --- Penalty for being stuck (didn't move from previous position, even if it tried) ---
        # This is distinct from hitting an obstacle (which results in no movement)
        if self.uav_pos == prev_pos and new_uav_pos not in self.fixed_obstacles:
            reward -= 0.5 # A more significant penalty for being completely stuck (tune this)

        # Ensure battery and time don't go below zero for state representation
        self.battery = max(0.0, self.battery)
        self.time_left = max(0.0, self.time_left)

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
            'step_reward_breakdown': { # Added for debugging specific rewards
                'exploration_reward': 0.1 if self.coverage_map[self.uav_pos] == 1.0 and prev_pos != self.uav_pos and self.victim_map[self.uav_pos] != 1.0 else 0, # simplified to exclude victims
                'victim_reward': 5.0 if self.victim_map[self.uav_pos] == 0.0 and prev_pos != self.uav_pos else 0, # if victim was there and is now 0
                'revisit_penalty': -0.02 if self.uav_pos != prev_pos and self.coverage_map[self.uav_pos] == 1.0 and new_uav_pos not in self.fixed_obstacles else 0,
                'obstacle_penalty': -2.0 if new_uav_pos in self.fixed_obstacles else 0,
                'stuck_penalty': -0.5 if self.uav_pos == prev_pos and new_uav_pos not in self.fixed_obstacles else 0,
                'total_step_specific_reward': step_reward # sum of the above specific rewards
            },
            'total_reward_this_step': reward # Final combined reward
        }

# ... (rest of your classes and training loop) ...

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

        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        log_probs = torch.stack(log_probs).to(device)
        values = torch.cat(values).squeeze().to(device)
        advantage = returns - values

        dist_entropy = -torch.sum(torch.exp(log_probs) * log_probs).mean()

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + critic_loss - 0.01 * dist_entropy  # Entropy bonus

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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

        stuck_steps = 0
        last_pos_for_stuck_check = env.uav_pos # Use a separate variable for stuck check
        
        f2_debug, f3_debug, f4_debug = 0, 0, 0
        current_step_reward_debug = 0 # To capture the step_reward from info

        for step in range(max_steps):
            # This 'stuck_steps' logic might interfere with the new penalties
            # It's usually better to let the reward function handle getting stuck.
            # However, if you want an explicit exploration nudge after being stuck, keep it.
            if env.uav_pos == last_pos_for_stuck_check:
                stuck_steps += 1
            else:
                stuck_steps = 0
            last_pos_for_stuck_check = env.uav_pos

            # If stuck for 2+ steps, sample a random action to break out
            if stuck_steps >= 2:
                action = env.action_space.sample()
                # We still need a log_prob and value from the model for PPO update
                # So, we pass the state through the model, but override action
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                logits, value = model(state_tensor)
                probs = torch.softmax(logits, dim=1)
                dist = torch.distributions.Categorical(probs)
                log_prob = dist.log_prob(torch.tensor(action).to(device)) # Log prob of the sampled action
                
                stuck_steps = 0 # Reset stuck counter after forced action
            else:
                action, log_prob, value = agent.select_action(state)

            next_state, reward, done, info = env.step(action)

            f2_debug = info['f2_energy']
            f3_debug = info['f3_time']
            f4_debug = info['f4_risk']
            current_step_reward_debug = info['total_reward_this_step'] # Get the new total reward

            records.append({
                'episode': episode,
                'step': step,
                'action': action,
                'reward': reward, # This is the immediate reward for the step
                'energy_used': info['energy_used'],
                'mission_time': info['mission_time'],
                'risk_score': info['risk_score'],
                'f2_energy': info['f2_energy'],
                'f3_time': info['f3_time'],
                'f4_risk': info['f4_risk'],
                'exploration_reward': info['step_reward_breakdown']['exploration_reward'],
                'victim_reward': info['step_reward_breakdown']['victim_reward'],
                'revisit_penalty': info['step_reward_breakdown']['revisit_penalty'],
                'obstacle_penalty': info['step_reward_breakdown']['obstacle_penalty'],
                'stuck_penalty': info['step_reward_breakdown']['stuck_penalty'],
                'total_step_specific_reward': info['step_reward_breakdown']['total_step_specific_reward']
            })

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward) # Collect immediate rewards for PPO update
            total_reward += reward
            state = next_state

            if done:
                break

        agent.update(rewards, log_probs, values)
        # Printing total reward per episode and last step's component rewards
        print(f"Episode {episode+1}/{num_episodes} - Total Episode Reward: {total_reward:.4f} | Last Step F-scores: f2: {f2_debug:.4f}, f3: {f3_debug:.4f}, f4: {f4_debug:.4f} | Last Step Total Reward: {current_step_reward_debug:.4f}")


    os.makedirs("model_output", exist_ok=True)
    torch.save(model.state_dict(), "model_output/uav_ppo_cnn_model.pth")
    pd.DataFrame(records).to_csv("model_output/training_log.csv", index=False)

if __name__ == "__main__":
    train()