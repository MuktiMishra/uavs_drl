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

# --- Configuration ---
# Set device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Reproducibility (important for consistent results)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- DiagnosticUAVEnv: Custom Gym Environment ---
class DiagnosticUAVEnv(gym.Env):
    """
    A custom Gym environment for a UAV navigating a grid to rescue victims
    while avoiding obstacles and managing resources (battery, time, risk).
    """
    def __init__(self, debug=False):
        super(DiagnosticUAVEnv, self).__init__()
        self.debug = debug
        self.grid_size = (15, 15)
        self.grid_height, self.grid_width = self.grid_size

        # Observation space: 8 channels (coverage, victim, rescued, obstacle, UAV pos, battery, time, risk)
        self.channels = 8
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.channels, *self.grid_size), dtype=np.float32)
        # Action space: 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
        self.action_space = spaces.Discrete(4)
        self.max_steps = 150  # Maximum steps per episode

        # Fixed locations for victims and obstacles for consistent environment
        self.fixed_victims = [
            (0, 2), (1, 4), (2, 8), (3, 1), (4, 12), (5, 3), (6, 6), (7, 10),
            (8, 14), (9, 7), (10, 5), (11, 2), (12, 13), (13, 0), (14, 4)
        ]
        self.fixed_obstacles = [
            (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 7), (7, 8),
            (8, 9), (9, 10), (10, 11), (11, 12)
        ]

        self.total_victims = len(self.fixed_victims)
        self.base_risk_map = self._create_risk_map() # Base risk across the grid

        self.action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT'] # For debugging output

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

    def reset(self):
        """
        Resets the environment to an initial state for a new episode.
        Initializes UAV position, maps, and resource levels.
        """
        self.step_count = 0
        self.energy_used = 0.0
        self.mission_time = 0.0
        self.risk_score = 0.0
        self.victims_rescued = 0

        # Initialize environment maps
        self.coverage_map = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        self.victim_map = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        self.rescued_map = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        self.obstacle_map = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)

        # Choose a random starting position for the UAV, not on an obstacle or victim
        valid_positions = [(y, x) for y in range(self.grid_height) for x in range(self.grid_width)
                           if (y, x) not in self.fixed_victims and (y, x) not in self.fixed_obstacles]
        self.uav_pos = random.choice(valid_positions)

        # Populate static maps
        for (y, x) in self.fixed_victims:
            self.victim_map[y, x] = 1
        for (y, x) in self.fixed_obstacles:
            self.obstacle_map[y, x] = 1

        self.battery = 1.0 # Initial battery level (normalized)
        self.time_left = 1.0 # Initial time remaining (normalized)
        self.risk_map = self.base_risk_map.copy() # Current risk map

        # History to detect if the UAV is stuck
        self.position_history = deque(maxlen=5)
        self.position_history.append(self.uav_pos)

        # Reset reward components for debugging
        self.total_reward_components = {
            'victim_rescue': 0, 'exploration': 0, 'movement': 0,
            'energy_penalty': 0, 'risk_penalty': 0, 'stuck_penalty': 0,
            'time_penalty': 0, 'completion_bonus': 0
        }

        return self.get_state()

    def get_state(self):
        """
        Constructs the current observation (state) for the agent.
        This is a multi-channel 2D grid representing various aspects of the environment.
        """
        uav_layer = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        uav_layer[self.uav_pos] = 1.0 # UAV's current position

        battery_layer = np.full((self.grid_height, self.grid_width), self.battery, dtype=np.float32)
        time_layer = np.full((self.grid_height, self.grid_width), self.time_left, dtype=np.float32)

        coverage_normalized = self.coverage_map.copy() # Cells visited by UAV

        # Stack all relevant information into a single state tensor
        stacked = np.stack([
            coverage_normalized, # Channel 0: Where the UAV has been
            self.victim_map,     # Channel 1: Locations of all victims
            self.rescued_map,    # Channel 2: Which victims have been rescued
            self.obstacle_map,   # Channel 3: Locations of obstacles
            uav_layer,           # Channel 4: UAV's current position
            battery_layer,       # Channel 5: Current battery level (broadcasted across grid)
            time_layer,          # Channel 6: Remaining mission time (broadcasted across grid)
            self.risk_map        # Channel 7: Risk map
        ], axis=0)

        return stacked

    def step(self, action):
        """
        Applies an action to the environment and calculates the next state, reward,
        and whether the episode is done.
        """
        self.step_count += 1
        prev_pos = self.uav_pos # Store previous position for movement check

        # Determine new UAV position based on action
        y, x = self.uav_pos
        if action == 0 and y > 0: y -= 1  # UP
        elif action == 1 and y < self.grid_height - 1: y += 1 # DOWN
        elif action == 2 and x > 0: x -= 1  # LEFT
        elif action == 3 and x < self.grid_width - 1: x += 1 # RIGHT

        # Update UAV position if not moving into an obstacle
        if (y, x) not in self.fixed_obstacles:
            self.uav_pos = (y, x)

        self.position_history.append(self.uav_pos) # Update position history

        # Update coverage map: Mark current UAV position as covered
        was_new_area = self.coverage_map[self.uav_pos] == 0
        self.coverage_map[self.uav_pos] = 1.0

        # Check for victim rescue at current position
        victim_rescued_this_step = False
        if self.victim_map[self.uav_pos] == 1 and self.rescued_map[self.uav_pos] == 0:
            self.rescued_map[self.uav_pos] = 1
            self.victims_rescued += 1
            victim_rescued_this_step = True

        # Simplified energy consumption model
        energy_consumed = 0.003  # Base energy cost per step
        if self.uav_pos != prev_pos:
            energy_consumed += 0.002  # Additional energy for movement
        if victim_rescued_this_step:
            energy_consumed += 0.005  # Additional energy for rescue operation

        self.energy_used += energy_consumed
        self.battery = max(0, self.battery - energy_consumed) # Decrease battery

        # Time progression
        time_step = 1.0 / self.max_steps # Normalized time per step
        self.mission_time += time_step
        self.time_left = max(0, self.time_left - time_step)

        # Risk accumulation
        self.risk_score += self.risk_map[self.uav_pos] * 0.01 # Accumulate risk based on current cell

        # Determine if the episode is terminated
        done = (self.step_count >= self.max_steps or # Max steps reached
                self.battery <= 0 or                 # Battery depleted
                self.time_left <= 0 or               # Time ran out
                self.victims_rescued == self.total_victims) # All victims rescued

        # Calculate the reward for this step
        reward = self._calculate_detailed_reward(victim_rescued_this_step, prev_pos, was_new_area)

        next_state = self.get_state() # Get the next observation

        # Information dictionary for debugging and logging
        info = {
            'energy_used': self.energy_used,
            'mission_time': self.mission_time,
            'risk_score': self.risk_score,
            'battery': self.battery,
            'time_left': self.time_left,
            'step': self.step_count,
            'victims_rescued': self.victims_rescued,
            'total_victims': self.total_victims,
            'victim_rescued_this_step': victim_rescued_this_step,
            'coverage_ratio': np.sum(self.coverage_map) / (self.grid_height * self.grid_width),
            'reward_components': self.total_reward_components.copy(), # Snapshot of total accumulated reward components
            'action_taken': self.action_names[action],
            'position': self.uav_pos
        }

        return next_state, reward, done, info

    def _calculate_detailed_reward(self, victim_rescued, prev_pos, was_new_area):
        """
        Calculates the reward for the current step based on various factors.
        Tracks individual reward components for analysis.
        """
        reward_components = {
            'victim_rescue': 0, 'exploration': 0, 'movement': 0,
            'energy_penalty': 0, 'risk_penalty': 0, 'stuck_penalty': 0,
            'time_penalty': 0, 'completion_bonus': 0
        }

        # 1. Victim rescue reward (most significant positive reward)
        if victim_rescued:
            reward_components['victim_rescue'] = 20.0

        # 2. Exploration reward: Encourage visiting new cells
        if was_new_area:
            reward_components['exploration'] = 1.0

        # 3. Movement reward/penalty: Encourage moving, mildly penalize staying still
        if self.uav_pos != prev_pos:
            reward_components['movement'] = 0.1
        else:
            reward_components['movement'] = -0.1

        # 4. Energy penalty: Mild penalty for using energy
        reward_components['energy_penalty'] = -0.5 * (0.003 + (0.002 if self.uav_pos != prev_pos else 0) + (0.005 if victim_rescued else 0))

        # 5. Risk penalty: Penalize being in high-risk areas
        reward_components['risk_penalty'] = -self.risk_map[self.uav_pos] * 0.1

        # 6. Stuck penalty: Penalize repeatedly staying in the same few positions
        if len(set(self.position_history)) <= 2 and self.step_count > 5: # Only apply after initial movement
            reward_components['stuck_penalty'] = -0.5

        # 7. Time penalty: Very mild penalty for each step to encourage efficiency
        reward_components['time_penalty'] = -0.01

        # 8. Mission completion bonus: Large bonus for rescuing all victims,
        # with an efficiency bonus based on remaining battery and time.
        if self.victims_rescued == self.total_victims and self.victims_rescued > 0:
            efficiency_bonus = (self.battery + self.time_left) * 10
            reward_components['completion_bonus'] = 50.0 + efficiency_bonus

        # Update total accumulated reward components for episode-end debugging
        for key, value in reward_components.items():
            self.total_reward_components[key] += value

        total_reward = sum(reward_components.values())
        return total_reward

    def print_debug_info(self):
        """
        Prints detailed information about the episode's outcome,
        useful for debugging and understanding agent behavior.
        """
        print(f"Episode completed in {self.step_count} steps")
        print(f"Victims rescued: {self.victims_rescued}/{self.total_victims}")
        print(f"Coverage: {np.sum(self.coverage_map)}/{self.grid_height * self.grid_width} "
              f"({np.sum(self.coverage_map)/(self.grid_height * self.grid_width)*100:.1f}%)")
        print(f"Final position: {self.uav_pos}")
        print(f"Battery remaining: {self.battery:.3f}")
        print(f"Time remaining: {self.time_left:.3f}")
        print("Total reward components breakdown:")
        for component, value in self.total_reward_components.items():
            if abs(value) > 0.01: # Only print significant components
                print(f"    {component}: {value:.2f}")
        print("-" * 30)


# --- DiagnosticCNNActorCritic: Neural Network Model ---
class DiagnosticCNNActorCritic(nn.Module):
    """
    Convolutional Neural Network (CNN) based Actor-Critic model.
    It takes the multi-channel grid state as input and outputs policy logits
    (for action selection) and a value estimate (for state evaluation).
    """
    def __init__(self, input_channels=8, num_actions=4):
        super(DiagnosticCNNActorCritic, self).__init__()

        # Convolutional layers to process the grid state
        self.conv1 = nn.Conv2d(input_channels, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)

        self.flatten = nn.Flatten() # Flatten the output of conv layers

        # Fully connected layers
        # The input size to fc1 depends on the output of conv2 (channels * height * width)
        # For a 15x15 grid with 8 channels, it's 8 * 15 * 15
        self.fc1 = nn.Linear(8 * 15 * 15, 128)
        self.fc2 = nn.Linear(128, 64)

        # Policy head: outputs logits for each action
        self.policy_head = nn.Linear(64, num_actions)
        # Value head: outputs a single scalar value estimate for the state
        self.value_head = nn.Linear(64, 1)

        self._init_weights() # Custom weight initialization

    def _init_weights(self):
        """
        Initializes weights using orthogonal initialization, which can improve
        training stability for reinforcement learning models.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5) # Orthogonal initialization with smaller gain
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0) # Initialize biases to zero

    def forward(self, x):
        """
        Defines the forward pass of the network.
        """
        x = torch.relu(self.conv1(x)) # Apply ReLU activation after conv layers
        x = torch.relu(self.conv2(x))
        x = self.flatten(x) # Flatten for FC layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        policy_logits = self.policy_head(x) # Raw logits for policy
        value = self.value_head(x) # Value estimate

        return policy_logits, value


# --- DiagnosticPPOAgent: Proximal Policy Optimization Agent ---
class DiagnosticPPOAgent:
    """
    Implements the Proximal Policy Optimization (PPO) algorithm for training
    the DiagnosticCNNActorCritic model.
    """
    def __init__(self, model, lr=1e-4, gamma=0.99, eps_clip=0.2, k_epochs=4, entropy_coef=0.02):
        self.model = model # The actor-critic network
        self.optimizer = optim.Adam(model.parameters(), lr=lr) # Adam optimizer
        self.gamma = gamma       # Discount factor for future rewards
        self.eps_clip = eps_clip # Clipping parameter for PPO
        self.k_epochs = k_epochs # Number of epochs to update the policy
        self.entropy_coef = entropy_coef # Coefficient for entropy regularization

        self.clear_memory() # Initialize replay buffer

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
        with torch.no_grad(): # No gradient calculation needed for action selection
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device) # Add batch dimension
            logits, value = self.model(state_tensor)

            probs = torch.softmax(logits, dim=1) # Convert logits to probabilities
            dist = torch.distributions.Categorical(probs) # Create a categorical distribution
            action = dist.sample() # Sample an action from the distribution
            log_prob = dist.log_prob(action) # Get the log probability of the sampled action

            return action.item(), log_prob.item(), value.item() # Return scalar values

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
        Calculates and returns separate loss components for analysis.
        """
        if len(self.states) == 0:
            return 0.0, 0.0, 0.0, 0.0 # Return zeros if no data to update

        # Convert collected experiences to PyTorch tensors
        states = torch.tensor(np.array(self.states), dtype=torch.float32).to(device)
        actions = torch.tensor(self.actions, dtype=torch.long).to(device)
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32).to(device)
        old_values = torch.tensor(self.values, dtype=torch.float32).to(device)

        returns = self._calculate_returns() # Calculate discounted cumulative rewards (returns)
        advantages = returns - old_values # Calculate advantages (how much better an action was than expected)

        # Normalize advantages for stability
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0

        # Perform K epochs of optimization
        for _ in range(self.k_epochs):
            logits, values = self.model(states) # Get current policy logits and value estimates
            probs = torch.softmax(logits, dim=1)
            dist = torch.distributions.Categorical(probs)

            new_log_probs = dist.log_prob(actions) # Log probability of actions under current policy
            entropy = dist.entropy() # Entropy of the policy distribution

            # PPO policy clipping objective
            ratio = torch.exp(new_log_probs - old_log_probs) # Importance sampling ratio
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean() # Negative sign for gradient ascent

            # Value function loss (MSE between predicted values and returns)
            value_loss = nn.MSELoss()(values.squeeze(), returns)

            # Entropy regularization term: encourages exploration
            entropy_loss = -self.entropy_coef * entropy.mean()

            # Total loss for backpropagation
            loss = policy_loss + 0.5 * value_loss + entropy_loss

            # Accumulate loss components for tracking
            total_actor_loss += policy_loss.item()
            total_critic_loss += value_loss.item()
            total_entropy += entropy.mean().item()

            # Backpropagation and optimization
            self.optimizer.zero_grad() # Clear previous gradients
            loss.backward() # Compute gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5) # Clip gradients to prevent explosion
            self.optimizer.step() # Update model parameters

        self.clear_memory() # Clear memory after update
        # Return average loss components over k_epochs
        avg_loss = (total_actor_loss + total_critic_loss) / self.k_epochs
        return (avg_loss,
                total_actor_loss / self.k_epochs,
                total_critic_loss / self.k_epochs,
                total_entropy / self.k_epochs)

    def _calculate_returns(self):
        """
        Calculates discounted cumulative rewards (returns) for each step
        in the collected trajectory.
        """
        returns = []
        discounted_sum = 0

        # Iterate backwards through rewards to calculate discounted returns
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done: # If episode ended, reset discounted sum
                discounted_sum = 0
            discounted_sum = reward + self.gamma * discounted_sum # Bellman equation
            returns.insert(0, discounted_sum) # Insert at beginning to maintain original order

        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        return returns


# --- Training Function ---
def detailed_train_with_loss_tracking():
    """
    Main training function. Sets up the environment, model, and agent,
    then runs the training loop, logs metrics, and saves results.
    """
    env = DiagnosticUAVEnv(debug=True) # Initialize environment with debug mode
    model = DiagnosticCNNActorCritic(input_channels=8).to(device) # Initialize model
    agent = DiagnosticPPOAgent(model, lr=1e-4, entropy_coef=0.02) # Initialize PPO agent

    # Training parameters
    num_episodes = 3000
    batch_size = 16 # Number of episodes to collect before updating the policy
    print_interval = 50 # How often to print progress
    # debug_interval = 200 # Removed, not used in this function

    # Lists to store metrics for plotting and analysis
    episode_rewards = []
    episode_victims = []
    episode_coverage = []
    episode_steps = []
    training_losses = []
    actor_losses = []
    critic_losses = []
    entropy_values = []

    print("Starting diagnostic training with loss tracking...")

    episode_batch = 0 # Counter for episodes in current batch
    best_victims = 0 # Track best performance (not explicitly used but good to have)

    for episode in range(num_episodes):
        state = env.reset() # Reset environment for new episode
        episode_reward = 0
        steps = 0

        while True:
            action, log_prob, value = agent.select_action(state) # Agent selects action
            next_state, reward, done, info = env.step(action) # Environment takes a step

            agent.store_transition(state, action, reward, log_prob, value, done) # Store experience

            episode_reward += reward
            steps += 1
            state = next_state # Move to next state

            if done: # Break if episode terminates
                break

        # Log episode-specific metrics
        episode_rewards.append(episode_reward)
        episode_victims.append(info['victims_rescued'])
        episode_coverage.append(info['coverage_ratio'])
        episode_steps.append(steps)
        episode_batch += 1

        # Perform policy update if batch size is reached
        if episode_batch >= batch_size:
            avg_loss, actor_loss, critic_loss, entropy = agent.update_with_loss_tracking()
            training_losses.append(avg_loss)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            entropy_values.append(entropy)
            episode_batch = 0 # Reset batch counter

        # Print training progress
        if (episode + 1) % print_interval == 0:
            recent_episodes = min(print_interval, len(episode_rewards))
            avg_reward = np.mean(episode_rewards[-recent_episodes:])
            avg_victims = np.mean(episode_victims[-recent_episodes:])

            print(f"Episode {episode+1}/{num_episodes}")
            print(f"    Avg Reward (last {recent_episodes}): {avg_reward:.2f}")
            print(f"    Avg Victims (last {recent_episodes}): {avg_victims:.2f}/{env.total_victims}")
            if training_losses: # Only print losses if an update has occurred
                print(f"    Recent Losses - Total: {training_losses[-1]:.4f}, Actor: {actor_losses[-1]:.4f}, Critic: {critic_losses[-1]:.4f}, Entropy: {entropy_values[-1]:.4f}")
            print()

            # Optionally print debug info for the current episode (can be verbose)
            # if (episode + 1) % debug_interval == 0:
            #    env.print_debug_info() # Detailed breakdown of the last episode's rewards

    # --- Save Results ---
    os.makedirs("model_output", exist_ok=True)
    torch.save(model.state_dict(), "model_output/diagnostic_uav_ppo_model_with_losses.pth")
    print(f"Model saved to model_output/diagnostic_uav_ppo_model_with_losses.pth")

    # Save detailed training results to a CSV file
    results_df = pd.DataFrame({
        'episode': range(len(episode_rewards)),
        'reward': episode_rewards,
        'victims_rescued': episode_victims,
        'coverage_ratio': episode_coverage,
        'steps': episode_steps,
        # Losses might be shorter if batch_size > 1 and num_episodes is not a multiple of batch_size
        'total_loss': training_losses + [np.nan] * (len(episode_rewards) - len(training_losses)),
        'actor_loss': actor_losses + [np.nan] * (len(episode_rewards) - len(actor_losses)),
        'critic_loss': critic_losses + [np.nan] * (len(episode_rewards) - len(critic_losses)),
        'entropy': entropy_values + [np.nan] * (len(episode_rewards) - len(entropy_values))
    })
    results_df.to_csv("model_output/diagnostic_training_results_with_losses.csv", index=False)
    print(f"Training results saved to model_output/diagnostic_training_results_with_losses.csv")

    return results_df

# --- Plotting Function ---
def plot_training_results(results_path="model_output/diagnostic_training_results_with_losses.csv",
                          start_episode=0, end_episode=None):
    """
    Loads training results from a CSV and generates plots for reward, loss components,
    and performance metrics.
    """
    if not os.path.exists(results_path):
        print(f"Error: Results file not found at {results_path}. Please run training first.")
        return

    df = pd.read_csv(results_path)

    # Create a dummy environment instance to get total_victims for plotting
    env = DiagnosticUAVEnv(debug=False)

    # Filter episodes for plotting range
    if end_episode is None:
        end_episode = len(df)
    df = df[(df['episode'] >= start_episode) & (df['episode'] < end_episode)] # Use < for end_episode exclusivity

    # Create subplots
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(12, 25)) # Added an extra subplot for total loss

    # Plot 1: Reward
    ax1.plot(df['episode'], df['reward'], label='Episode Reward', color='blue', alpha=0.6)
    ax1.set_title(f'Training Rewards (Episodes {start_episode}-{end_episode})')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(True)
    ax1.legend()

    # Plot 2: Total Loss
    ax2.plot(df['episode'], df['total_loss'], label='Total Loss', color='red')
    ax2.set_title(f'Total Loss (Episodes {start_episode}-{end_episode})')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    # Plot 3: Critic Loss
    ax3.plot(df['episode'], df['critic_loss'], label='Critic Loss', color='purple')
    ax3.set_title(f'Critic Loss (Episodes {start_episode}-{end_episode})')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.grid(True)

    # Plot 4: Actor Loss
    ax4.plot(df['episode'], df['actor_loss'], label='Actor Loss', color='green')
    ax4.set_title(f'Actor Loss (Episodes {start_episode}-{end_episode})')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Loss')
    ax4.legend()
    ax4.grid(True)

    # Plot 5: Performance Metrics (Victims Rescued & Coverage)
    ax5.plot(df['episode'], df['victims_rescued'], label='Victims Rescued', color='orange')
    # Scale coverage ratio to the number of victims for better comparison on the same y-axis
    ax5.plot(df['episode'], df['coverage_ratio'] * (env.grid_height * env.grid_width),
             label=f'Coverage (scaled to {env.grid_height * env.grid_width} cells)', color='cyan', alpha=0.5)
    ax5.set_title(f'Performance Metrics (Episodes {start_episode}-{end_episode})')
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Count / Ratio')
    ax5.legend()
    ax5.grid(True)

    plt.tight_layout() # Adjust layout to prevent overlapping titles/labels
    os.makedirs("model_output/plots", exist_ok=True) # Ensure plots directory exists
    plot_filename = f"model_output/plots/training_metrics_{start_episode}_to_{end_episode}.png"
    plt.savefig(plot_filename)
    print(f"Plots saved to {plot_filename}")
    plt.show() # Display the plots


# --- Simulation Function (for checking trained model behavior) ---
def run_simulation_and_get_trajectory(model_path="model_output/diagnostic_uav_ppo_model_with_losses.pth"):
    """
    Runs a single simulation episode using the trained model and collects the trajectory data.
    This function will load the saved model and use it to interact with the environment.
    """
    env = DiagnosticUAVEnv(debug=True) # Use debug mode to print detailed info at end
    model = DiagnosticCNNActorCritic(input_channels=8).to(device)

    # Load the trained model weights
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() # Set model to evaluation mode (important for inference)
        print(f"Loaded model from {model_path} for simulation.")
    else:
        print(f"Warning: Model not found at {model_path}. Running simulation with an untrained model.")
        model.eval() # Still set to eval mode even if untrained

    agent = DiagnosticPPOAgent(model) # Create an agent instance (only needs model for selection)

    trajectory = []
    state = env.reset()
    done = False
    
    # Store initial state in trajectory
    initial_step_data = {
        'uav_pos': env.uav_pos,
        'victim_map': env.victim_map.tolist(),
        'rescued_map': env.rescued_map.tolist(),
        'obstacle_map': env.obstacle_map.tolist(),
        'grid_height': env.grid_height,
        'grid_width': env.grid_width,
        'step_count': env.step_count,
        'victims_rescued': env.victims_rescued,
        'total_victims': env.total_victims
    }
    trajectory.append(initial_step_data)

    print("Running simulation episode...")
    while not done:
        action, _, _ = agent.select_action(state) # Select action using the trained model
        next_state, reward, done, info = env.step(action) # Take a step in the environment

        # Collect relevant data for visualization at the current step
        current_step_data = {
            'uav_pos': info['position'], # Use info['position'] for accuracy if UAV didn't move
            'victim_map': env.victim_map.tolist(),
            'rescued_map': env.rescued_map.tolist(),
            'obstacle_map': env.obstacle_map.tolist(),
            'grid_height': env.grid_height,
            'grid_width': env.grid_width,
            'step_count': info['step'],
            'victims_rescued': info['victims_rescued'],
            'total_victims': info['total_victims']
        }
        trajectory.append(current_step_data)

        state = next_state # Update state

        if info['step'] >= env.max_steps and not done: # Ensure termination if max_steps reached but 'done' not true (edge case)
            done = True
            print("Simulation terminated early: Reached max_steps.")

    print(f"Simulation completed. Total steps: {len(trajectory)-1}.") # -1 because initial state is included
    env.print_debug_info() # Print detailed info about the simulation run

    return trajectory # Return the collected trajectory


# --- Main Execution Block ---
if __name__ == "__main__":
    # Create the model_output directory if it doesn't exist
    os.makedirs("model_output", exist_ok=True)

    # Check if training results already exist. If not, run training.
    # This prevents retraining every time you run the script if you just want to plot or simulate.
    results_file = "model_output/diagnostic_training_results_with_losses.csv"
    model_file = "model_output/diagnostic_uav_ppo_model_with_losses.pth"

    if not os.path.exists(results_file) or not os.path.exists(model_file):
        print("Training results or model not found. Starting training...")
        detailed_train_with_loss_tracking()
    else:
        print("Training results and model found. Skipping training.")

    # Always plot the results after training (or if they already exist)
    print("\nPlotting training results...")
    plot_training_results() # Plots all episodes by default

    # You can specify a range for plotting, e.g., plot_training_results(start_episode=1000, end_episode=2000)

    # Run a simulation episode using the (trained) model and get the trajectory
    print("\nRunning a simulation episode to inspect behavior...")
    simulation_trajectory = run_simulation_and_get_trajectory()

    # The 'simulation_trajectory' list now contains a detailed log of each step
    # of the UAV during a single run. You can further process this data,
    # for example, to build a visualization with external tools or libraries
    # (like the HTML/JS example provided previously, but this Python script doesn't visualize).
    print(f"\nSimulation trajectory collected. It contains {len(simulation_trajectory)} steps.")
    print("Example of first step in trajectory:", simulation_trajectory[0])