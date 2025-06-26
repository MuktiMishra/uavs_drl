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

# Re-define the DiagnosticUAVEnv class (must be identical to training)
class DiagnosticUAVEnv(gym.Env):
    def __init__(self, debug=False):
        super(DiagnosticUAVEnv, self).__init__()
        self.debug = debug
        self.grid_size = (15, 15)
        self.grid_height, self.grid_width = self.grid_size

        self.channels = 8
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.channels, *self.grid_size), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

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
        self.energy_used = 0.0
        self.mission_time = 0.0
        self.risk_score = 0.0
        self.victims_rescued = 0

        self.coverage_map = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        self.victim_map = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        self.rescued_map = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        self.obstacle_map = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)

        valid_positions = [(y, x) for y in range(self.grid_height) for x in range(self.grid_width)
                           if (y, x) not in self.fixed_victims and (y, x) not in self.fixed_obstacles]
        self.uav_pos = random.choice(valid_positions)

        for (y, x) in self.fixed_victims:
            self.victim_map[y, x] = 1
        for (y, x) in self.fixed_obstacles:
            self.obstacle_map[y, x] = 1

        self.battery = 1.0
        self.time_left = 1.0
        self.risk_map = self.base_risk_map.copy()

        self.position_history = deque(maxlen=10)
        self.position_history.append(self.uav_pos)

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

        return self.get_state()

    def get_state(self):
        uav_layer = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        uav_layer[self.uav_pos] = 1.0

        battery_layer = np.full((self.grid_height, self.grid_width), self.battery, dtype=np.float32)
        time_layer = np.full((self.grid_height, self.grid_width), self.time_left, dtype=np.float32)

        coverage_normalized = self.coverage_map.copy()

        stacked = np.stack([
            coverage_normalized,
            self.victim_map,
            self.rescued_map,
            self.obstacle_map,
            uav_layer,
            battery_layer,
            time_layer,
            self.risk_map
        ], axis=0)

        return stacked

    def step(self, action):
        self.step_count += 1
        prev_pos = self.uav_pos

        y, x = self.uav_pos
        if action == 0 and y > 0: y -= 1
        elif action == 1 and y < self.grid_height - 1: y += 1
        elif action == 2 and x > 0: x -= 1
        elif action == 3 and x < self.grid_width - 1: x += 1

        if (y, x) not in self.fixed_obstacles:
            self.uav_pos = (y, x)

        self.position_history.append(self.uav_pos)

        was_new_area = self.coverage_map[self.uav_pos] == 0
        self.coverage_map[self.uav_pos] = 1.0

        victim_rescued = False
        if self.victim_map[self.uav_pos] == 1 and self.rescued_map[self.uav_pos] == 0:
            self.rescued_map[self.uav_pos] = 1
            self.victims_rescued += 1
            victim_rescued = True

        energy_consumed = 0.003
        if self.uav_pos != prev_pos:
            energy_consumed += 0.002
        if victim_rescued:
            energy_consumed += 0.005

        self.energy_used += energy_consumed
        self.battery = max(0, self.battery - energy_consumed)

        time_step = 1.0 / self.max_steps
        self.mission_time += time_step
        self.time_left = max(0, self.time_left - time_step)

        self.risk_score += self.risk_map[self.uav_pos] * 0.01

        done = (self.step_count >= self.max_steps or
                self.battery <= 0 or
                self.time_left <= 0 or
                self.victims_rescued == self.total_victims)

        reward = self._calculate_detailed_reward(victim_rescued, prev_pos, was_new_area)

        next_state = self.get_state()

        info = {
            'energy_used': self.energy_used,
            'mission_time': self.mission_time,
            'risk_score': self.risk_score,
            'battery': self.battery,
            'time_left': self.time_left,
            'step': self.step_count,
            'victims_rescued': self.victims_rescued,
            'total_victims': self.total_victims,
            'victim_rescued_this_step': victim_rescued,
            'coverage_ratio': np.sum(self.coverage_map) / (self.grid_height * self.grid_width),
            'reward_components': self.total_reward_components.copy(),
            'action_taken': self.action_names[action],
            'position': self.uav_pos
        }

        return next_state, reward, done, info

    def _calculate_detailed_reward(self, victim_rescued, prev_pos, was_new_area):
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

        if self.uav_pos != prev_pos:
            reward_components['movement'] = 0.2
        else:
            reward_components['movement'] = -0.2

        reward_components['energy_penalty'] = -self.energy_used * 0.4

        reward_components['risk_penalty'] = -self.risk_map[self.uav_pos] * 0.15

        if len(set(self.position_history)) <= 3 and self.step_count > self.max_steps * 0.1:
            reward_components['stuck_penalty'] = -2.0

        reward_components['time_penalty'] = -0.02

        if self.victims_rescued == self.total_victims:
            efficiency_bonus = (self.battery * 20.0 + self.time_left * 20.0)
            reward_components['completion_bonus'] = 100.0 + efficiency_bonus

        for key, value in reward_components.items():
            self.total_reward_components[key] += value

        total_reward = sum(reward_components.values())
        return total_reward

    def print_debug_info(self):
        print(f"Episode completed in {self.step_count} steps")
        print(f"Victims rescued: {self.victims_rescued}/{self.total_victims}")
        print(f"Coverage: {np.sum(self.coverage_map)}/{self.grid_height * self.grid_width} ({np.sum(self.coverage_map)/(self.grid_height * self.grid_width)*100:.1f}%)")
        print(f"Final position: {self.uav_pos}")
        print(f"Battery remaining: {self.battery:.3f}")
        print(f"Time remaining: {self.time_left:.3f}")
        print("Reward components:")
        for component, value in self.total_reward_components.items():
            if abs(value) > 0.01:
                print(f"   {component}: {value:.2f}")
        print()

# Re-define the DiagnosticCNNActorCritic class (must be identical to training)
class DiagnosticCNNActorCritic(nn.Module):
    def __init__(self, input_channels=8, num_actions=4):
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

def visualize_episode(env, path_history, episode_num, total_reward, victims_rescued, grid_size, fixed_obstacles, fixed_victims, rescued_victims):
    fig, ax = plt.subplots(figsize=(grid_size[1], grid_size[0]))

    # Draw grid lines
    ax.set_xticks(np.arange(-0.5, grid_size[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_size[0], 1), minor=True)
    ax.grid(which='minor', color='lightgray', linestyle='-', linewidth=1)
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot obstacles (black squares)
    for (oy, ox) in fixed_obstacles:
        rect = patches.Rectangle((ox - 0.5, oy - 0.5), 1, 1, linewidth=1, edgecolor='black', facecolor='black', alpha=0.8)
        ax.add_patch(rect)

    # Plot victim locations (red 'X' for unrescued, green circle for rescued)
    for (vy, vx) in fixed_victims:
        if (vy, vx) in rescued_victims:
            # Rescued victim: Green circle
            circle = patches.Circle((vx, vy), radius=0.3, color='green', fill=True, alpha=0.7)
            ax.add_patch(circle)
            ax.text(vx, vy, 'R', color='white', ha='center', va='center', fontsize=10, fontweight='bold')
        else:
            # Unrescued victim: Red X
            ax.plot(vx, vy, 'rx', markersize=10, markeredgewidth=2)

    # Plot UAV path
    path_y = [pos[0] for pos in path_history]
    path_x = [pos[1] for pos in path_history]
    ax.plot(path_x, path_y, 'b-o', markersize=3, linewidth=1, alpha=0.7, label='UAV Path')

    # Plot start and end points
    ax.plot(path_history[0][1], path_history[0][0], 'gs', markersize=8, label='Start') # Green square for start
    ax.plot(path_history[-1][1], path_history[-1][0], 'bo', markersize=8, label='End') # Blue circle for end

    ax.set_xlim(-0.5, grid_size[1] - 0.5)
    ax.set_ylim(-0.5, grid_size[0] - 0.5)
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis() # Invert Y-axis to match (row, col) indexing where row 0 is at the top

    ax.set_title(f"Episode {episode_num} - Total Reward: {total_reward:.2f}, Victims Rescued: {victims_rescued}/{env.total_victims}")
    ax.legend(loc='lower left', bbox_to_anchor=(1, 0)) # Move legend outside
    plt.tight_layout()

    plot_dir = "model_output/test_plots"
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, f"test_episode_{episode_num:03d}.png"))
    plt.close()

def test_model(model_path="model_output/diagnostic_uav_ppo_model_BEST.pth", num_test_episodes=10):
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
        path_history = []
        rescued_victims_this_episode = set()
        
        # Store initial position
        path_history.append(env.uav_pos)

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
            path_history.append(env.uav_pos)

            if info['victim_rescued_this_step']:
                rescued_victims_this_episode.add(info['position'])

        env.print_debug_info() # Print detailed info for each test episode

        visualize_episode(env, path_history, i+1, episode_reward, info['victims_rescued'], env.grid_size, env.fixed_obstacles, env.fixed_victims, rescued_victims_this_episode)

        test_results.append({
            'episode': i + 1,
            'total_reward': episode_reward,
            'victims_rescued': info['victims_rescued'],
            'coverage_ratio': info['coverage_ratio'],
            'steps_taken': info['step'],
            'battery_final': info['battery'],
            'time_left_final': info['time_left'],
            'risk_score_final': info['risk_score']
        })

    results_df = pd.DataFrame(test_results)
    results_csv_path = "model_output/test_results_IMPROVED_v2.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nTest results saved to {results_csv_path}")
    print("\n--- Testing complete ---")

    # Overall average results
    print("\nOverall Average Test Results:")
    print(f"  Avg Total Reward: {results_df['total_reward'].mean():.2f}")
    print(f"  Avg Victims Rescued: {results_df['victims_rescued'].mean():.2f}/{env.total_victims}")
    print(f"  Avg Coverage Ratio: {results_df['coverage_ratio'].mean():.3f}")
    print(f"  Avg Steps Taken: {results_df['steps_taken'].mean():.1f}")
    print(f"  Avg Final Battery: {results_df['battery_final'].mean():.3f}")
    print(f"  Avg Final Time Left: {results_df['time_left_final'].mean():.3f}")
    print(f"  Avg Final Risk Score: {results_df['risk_score_final'].mean():.3f}")


if __name__ == "__main__":
    # Specify the path to your trained model
    # Use the 'BEST' model if available, otherwise the 'final' one
    best_model_path = "model_output/diagnostic_uav_ppo_model_BEST_v2.pth"
    final_model_path = "model_output/diagnostic_uav_ppo_model_final_v2.pth"

    model_to_test = ""
    if os.path.exists(best_model_path):
        model_to_test = best_model_path
        print(f"Using best trained model: {model_to_test}")
    elif os.path.exists(final_model_path):
        model_to_test = final_model_path
        print(f"Using final trained model: {model_to_test}")
    else:
        print("No trained model found. Please run the training script first.")
        exit()

    # Number of episodes to run for testing
    num_episodes_to_test = 5

    test_model(model_path=model_to_test, num_test_episodes=num_episodes_to_test)