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

class DiagnosticUAVEnv(gym.Env):
    def __init__(self, debug=False):
        super(DiagnosticUAVEnv, self).__init__()
        self.debug = debug
        self.grid_size = (15, 15)
        self.grid_height, self.grid_width = self.grid_size

        self.channels = 8
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.channels, *self.grid_size), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.max_steps = 150  # Increased significantly
        
        # Reduced number of victims for initial testing
        self.fixed_victims = [
            (0, 2), (1, 4), (2, 8), (3, 1), (4, 12), (5, 3), (6, 6), (7, 10),
            (8, 14), (9, 7), (10, 5), (11, 2), (12, 13), (13, 0), (14, 4)
        ]  # 15 victims instead of 22
        
        # Reduced obstacles for easier navigation
        self.fixed_obstacles = [
            (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 7), (7, 8),
            (8, 9), (9, 10), (10, 11), (11, 12)
        ]  # 12 obstacles instead of 22
        
        self.total_victims = len(self.fixed_victims)
        self.base_risk_map = self._create_risk_map()
        
        # Action names for debugging
        self.action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']

    def _create_risk_map(self):
        """Create a milder risk map"""
        risk_map = np.random.rand(self.grid_height, self.grid_width) * 0.2 + 0.1
        
        # Add moderate risk near obstacles
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

        # Initialize maps
        self.coverage_map = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        self.victim_map = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        self.rescued_map = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        self.obstacle_map = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)

        # Random start position
        valid_positions = [(y, x) for y in range(self.grid_height) for x in range(self.grid_width)
                          if (y, x) not in self.fixed_victims and (y, x) not in self.fixed_obstacles]
        self.uav_pos = random.choice(valid_positions)

        # Set up maps
        for (y, x) in self.fixed_victims:
            self.victim_map[y, x] = 1
        for (y, x) in self.fixed_obstacles:
            self.obstacle_map[y, x] = 1

        self.battery = 1.0
        self.time_left = 1.0
        self.risk_map = self.base_risk_map.copy()
        
        self.position_history = deque(maxlen=5)
        self.position_history.append(self.uav_pos)
        
        # Debugging info
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

        # Move UAV
        y, x = self.uav_pos
        if action == 0 and y > 0: y -= 1
        elif action == 1 and y < self.grid_height - 1: y += 1
        elif action == 2 and x > 0: x -= 1
        elif action == 3 and x < self.grid_width - 1: x += 1
        
        if (y, x) not in self.fixed_obstacles:
            self.uav_pos = (y, x)
        
        self.position_history.append(self.uav_pos)

        # Update coverage map
        was_new_area = self.coverage_map[self.uav_pos] == 0
        self.coverage_map[self.uav_pos] = 1.0

        # Check for victim rescue
        victim_rescued = False
        if self.victim_map[self.uav_pos] == 1 and self.rescued_map[self.uav_pos] == 0:
            self.rescued_map[self.uav_pos] = 1
            self.victims_rescued += 1
            victim_rescued = True

        # Simplified energy model
        energy_consumed = 0.003  # Base energy per step
        if self.uav_pos != prev_pos:
            energy_consumed += 0.002  # Movement energy
        if victim_rescued:
            energy_consumed += 0.005  # Rescue energy
        
        self.energy_used += energy_consumed
        self.battery = max(0, self.battery - energy_consumed)

        # Time progression
        time_step = 1.0 / self.max_steps
        self.mission_time += time_step
        self.time_left = max(0, self.time_left - time_step)

        # Risk accumulation
        self.risk_score += self.risk_map[self.uav_pos] * 0.01

        # Check termination
        done = (self.step_count >= self.max_steps or 
                self.battery <= 0 or 
                self.time_left <= 0 or
                self.victims_rescued == self.total_victims)

        # Calculate reward with detailed tracking
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
        """Detailed reward calculation with component tracking"""
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
        
        # 1. Victim rescue reward (most important)
        if victim_rescued:
            reward_components['victim_rescue'] = 20.0
        
        # 2. Exploration reward
        if was_new_area:
            reward_components['exploration'] = 1.0
        
        # 3. Movement reward/penalty
        if self.uav_pos != prev_pos:
            reward_components['movement'] = 0.1
        else:
            reward_components['movement'] = -0.1
        
        # 4. Energy penalty (mild)
        reward_components['energy_penalty'] = -self.energy_used * 0.5
        
        # 5. Risk penalty (mild)
        reward_components['risk_penalty'] = -self.risk_map[self.uav_pos] * 0.1
        
        # 6. Stuck penalty
        if len(set(self.position_history)) <= 2:
            reward_components['stuck_penalty'] = -0.5
        
        # 7. Time penalty (very mild)
        reward_components['time_penalty'] = -0.01
        
        # 8. Mission completion bonus
        if self.victims_rescued == self.total_victims:
            efficiency_bonus = (self.battery + self.time_left) * 10
            reward_components['completion_bonus'] = 50.0 + efficiency_bonus
        
        # Update total components
        for key, value in reward_components.items():
            self.total_reward_components[key] += value
        
        total_reward = sum(reward_components.values())
        return total_reward

    def print_debug_info(self):
        """Print detailed episode information"""
        print(f"Episode completed in {self.step_count} steps")
        print(f"Victims rescued: {self.victims_rescued}/{self.total_victims}")
        print(f"Coverage: {np.sum(self.coverage_map)}/{self.grid_height * self.grid_width} ({np.sum(self.coverage_map)/(self.grid_height * self.grid_width)*100:.1f}%)")
        print(f"Final position: {self.uav_pos}")
        print(f"Battery remaining: {self.battery:.3f}")
        print(f"Time remaining: {self.time_left:.3f}")
        print("Reward components:")
        for component, value in self.total_reward_components.items():
            if abs(value) > 0.01:
                print(f"  {component}: {value:.2f}")
        print()

class DiagnosticCNNActorCritic(nn.Module):
    def __init__(self, input_channels=8, num_actions=4):
        super(DiagnosticCNNActorCritic, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8 * 15 * 15, 128)
        self.fc2 = nn.Linear(128, 64)
        
        self.policy_head = nn.Linear(64, num_actions)
        self.value_head = nn.Linear(64, 1)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)  # Smaller gain for stability
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        
        return policy_logits, value

class DiagnosticPPOAgent:
    def __init__(self, model, lr=1e-4, gamma=0.99, eps_clip=0.2, k_epochs=4, entropy_coef=0.02):
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
            return 0.0, 0.0, 0.0, 0.0
        
        states = torch.tensor(np.array(self.states), dtype=torch.float32).to(device)
        actions = torch.tensor(self.actions, dtype=torch.long).to(device)
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32).to(device)
        old_values = torch.tensor(self.values, dtype=torch.float32).to(device)
        
        returns = self._calculate_returns()
        advantages = returns - old_values
        
        # Normalize advantages
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        
        for _ in range(self.k_epochs):
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
            
            # Value loss
            value_loss = nn.MSELoss()(values.squeeze(), returns)
            
            # Entropy bonus
            entropy_loss = -self.entropy_coef * entropy.mean()
            
            # Total loss
            loss = policy_loss + 0.5 * value_loss + entropy_loss
            
            total_actor_loss += policy_loss.item()
            total_critic_loss += value_loss.item()
            total_entropy += entropy.mean().item()
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
        
        self.clear_memory()
        avg_loss = (total_actor_loss + total_critic_loss) / self.k_epochs
        return (avg_loss, 
                total_actor_loss / self.k_epochs, 
                total_critic_loss / self.k_epochs, 
                total_entropy / self.k_epochs)

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
    env = DiagnosticUAVEnv(debug=True)
    model = DiagnosticCNNActorCritic(input_channels=8).to(device)
    agent = DiagnosticPPOAgent(model, lr=1e-4, entropy_coef=0.02)
    
    # Training parameters
    num_episodes = 3000
    batch_size = 16
    print_interval = 50
    debug_interval = 200
    
    # Tracking metrics
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
        
        # Update agent and track losses
        if episode_batch >= batch_size:
            avg_loss, actor_loss, critic_loss, entropy = agent.update_with_loss_tracking()
            training_losses.append(avg_loss)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            entropy_values.append(entropy)
            episode_batch = 0
        
        # Print progress
        if (episode + 1) % print_interval == 0:
            recent_episodes = min(print_interval, len(episode_rewards))
            avg_reward = np.mean(episode_rewards[-recent_episodes:])
            avg_victims = np.mean(episode_victims[-recent_episodes:])
            
            print(f"Episode {episode+1}/{num_episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Victims: {avg_victims:.2f}/{env.total_victims}")
            if training_losses:
                print(f"  Recent Losses - Total: {training_losses[-1]:.4f}, Actor: {actor_losses[-1]:.4f}, Critic: {critic_losses[-1]:.4f}")
            print()
    
    # Save all results
    os.makedirs("model_output", exist_ok=True)
    torch.save(model.state_dict(), "model_output/diagnostic_uav_ppo_model_with_losses.pth")
    
    # Save more detailed results
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
    results_df.to_csv("model_output/diagnostic_training_results_with_losses.csv", index=False)
    
    return results_df

def plot_training_results(results_path="model_output/diagnostic_training_results_with_losses.csv", 
                         start_episode=0, end_episode=3000):
    # Load the data
    df = pd.read_csv(results_path)
    
    # Create environment instance
    env = DiagnosticUAVEnv(debug=False)
    
    # Filter episodes
    if end_episode is None:
        end_episode = len(df)
    df = df[(df['episode'] >= start_episode) & (df['episode'] <= end_episode)]
    
    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot reward (only selected episodes)
    ax1.plot(df['episode'], df['reward'], label='Episode Reward', color='blue', alpha=0.6)
    ax1.set_title(f'Training Rewards (Episodes {start_episode}-{end_episode})')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(True)
    
    # Plot losses (only selected episodes)
    # ax2.plot(df['episode'], df['total_loss'], label='Total Loss', color='red')
    ax2.plot(df['episode'], df['actor_loss'], label='Actor Loss', color='green')
    ax2.plot(df['episode'], df['critic_loss'], label='Critic Loss', color='purple')
    ax2.set_title(f'Training Losses (Episodes {start_episode}-{end_episode})')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    # Plot performance metrics (only selected episodes)
    ax3.plot(df['episode'], df['victims_rescued'], label='Victims Rescued', color='orange')
    ax3.plot(df['episode'], df['coverage_ratio'] * env.total_victims, 
             label='Coverage (scaled)', color='cyan', alpha=0.5)
    ax3.set_title(f'Performance Metrics (Episodes {start_episode}-{end_episode})')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Count')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"model_output/plots/training_metrics_{start_episode}_to_{end_episode}.png")
    plt.show()
if __name__ == "__main__":
    # Run training with loss tracking if needed
    if not os.path.exists("model_output/diagnostic_training_results_with_losses.csv"):
        print("Running training with loss tracking...")
        detailed_train_with_loss_tracking()
    
    # Plot the results
    plot_training_results()