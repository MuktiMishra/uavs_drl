# import torch
# import torch.nn.functional as F
# import torch.optim as optim
# import numpy as np
# from torch.distributions import Categorical
# import gym
# import numpy as np
# import importlib.util
# import sys
# import os
# import matplotlib.pyplot as plt


# from sar_env import SARGridEnv
# from ppo_model import CNNActorCritic
# # Load .config.py dynamically
# config_path = os.path.join(os.path.dirname(__file__), '.config.py')
# spec = importlib.util.spec_from_file_location("config", config_path)
# config_module = importlib.util.module_from_spec(spec)
# sys.modules["config"] = config_module
# spec.loader.exec_module(config_module)

# Config = config_module.Config

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def compute_gae(rewards, values, dones, gamma=Config.GAMMA, lam=Config.LAMBDA):
#     advantages = []
#     gae = 0
#     values = values + [0]
#     for step in reversed(range(len(rewards))):
#         delta = rewards[step] + gamma * values[step + 1] * (1 - dones[step]) - values[step]
#         gae = delta + gamma * lam * (1 - dones[step]) * gae
#         advantages.insert(0, gae)
#     return advantages

# def train():
#     env = SARGridEnv()
#     model = CNNActorCritic().to(device)
#     optimizer = optim.Adam(model.parameters(), lr=Config.LR)

#     timestep = 0
#     memory = {
#         "states": [],
#         "actions": [],
#         "log_probs": [],
#         "rewards": [],
#         "dones": [],
#         "values": []
#     }

#     state = env.reset()
#     episode_rewards = []
#     ep_reward = 0

#     while timestep < Config.TOTAL_TIMESTEPS:
#         # Collect rollout
#         for _ in range(Config.UPDATE_TIMESTEP):
#             state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)  # (1,7,15,15)
#             logits, value = model(state_tensor)
#             dist = Categorical(logits=logits)
#             action = dist.sample()
#             log_prob = dist.log_prob(action)

#             next_state, reward, done, _ = env.step(action.item())

#             memory["states"].append(state)
#             memory["actions"].append(action.item())
#             memory["log_probs"].append(log_prob.detach().cpu().numpy())
#             memory["rewards"].append(reward)
#             memory["dones"].append(done)
#             memory["values"].append(value.item())

#             state = next_state
#             ep_reward += reward
#             timestep += 1

#             if done:
#                 episode_rewards.append(ep_reward)
#                 print(f"Episode done: Reward: {ep_reward:.2f}, Timestep: {timestep}")
#                 state = env.reset()
#                 ep_reward = 0

#         # Compute advantage and returns
#         advantages = compute_gae(memory["rewards"], memory["values"], memory["dones"])
#         returns = [adv + val for adv, val in zip(advantages, memory["values"])]

#         # Convert memory to tensors
#         states = torch.FloatTensor(memory["states"]).to(device)
#         actions = torch.LongTensor(memory["actions"]).to(device)
#         old_log_probs = torch.FloatTensor(memory["log_probs"]).to(device)
#         returns = torch.FloatTensor(returns).to(device)
#         advantages = torch.FloatTensor(advantages).to(device)
#         advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

#         # PPO Update
#         for _ in range(Config.EPOCHS):
#             logits, values = model(states)
#             dist = Categorical(logits=logits)
#             entropy = dist.entropy().mean()
#             new_log_probs = dist.log_prob(actions)

#             ratio = (new_log_probs - old_log_probs).exp()
#             surr1 = ratio * advantages
#             surr2 = torch.clamp(ratio, 1 - Config.CLIP_EPS, 1 + Config.CLIP_EPS) * advantages
#             policy_loss = -torch.min(surr1, surr2).mean()

#             value_loss = F.mse_loss(values, returns)

#             loss = policy_loss + Config.VALUE_COEF * value_loss - Config.ENTROPY_COEF * entropy

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#         # Clear memory
#         for key in memory:
#             memory[key] = []

#         # Print average reward every update
#         if len(episode_rewards) > 0:
#             avg_reward = np.mean(episode_rewards[-10:])
#             print(f"Timestep: {timestep}, Avg Reward (last 10 eps): {avg_reward:.2f}")

# if __name__ == "__main__":
#     train()
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sar_env import SARGridEnv
from ppo_model import CNNActorCritic
import importlib.util
import os
import sys

# Load .config.py dynamically
config_path = os.path.join(os.path.dirname(__file__), '.config.py')
spec = importlib.util.spec_from_file_location("config", config_path)
config_module = importlib.util.module_from_spec(spec)
sys.modules["config"] = config_module
spec.loader.exec_module(config_module)
Config = config_module.Config

# Hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_UPDATES = Config.TOTAL_TIMESTEPS // Config.UPDATE_TIMESTEP

# Initialize environment and policy
env = SARGridEnv()
policy = CNNActorCritic().to(DEVICE)  # NO args here since your class uses Config internally
optimizer = optim.Adam(policy.parameters(), lr=Config.LR)

episode_rewards = []
coverage_progress = []

for update in range(NUM_UPDATES):
    obs = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    log_probs = []
    values = []
    rewards = []
    masks = []
    states = []
    actions = []

    total_reward = 0
    coverage_vals = []

    for _ in range(Config.UPDATE_TIMESTEP):
        logits, value = policy(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()

        next_obs, reward, done, _ = env.step(action.item())
        total_reward += reward
        coverage_vals.append(np.sum(env.coverage_map) / (env.grid_h * env.grid_w))

        log_prob = dist.log_prob(action)

        log_probs.append(log_prob)
        values.append(value)
        rewards.append(torch.tensor([reward], dtype=torch.float32).to(DEVICE))
        masks.append(torch.tensor([1 - done], dtype=torch.float32).to(DEVICE))
        states.append(obs)
        actions.append(action)

        if done:
            break

        obs = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # Compute returns and advantages
    _, next_value = policy(obs)
    returns = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + Config.GAMMA * next_value * masks[i] - values[i]
        gae = delta + Config.GAMMA * Config.LAMBDA * masks[i] * gae
        returns.insert(0, gae + values[i])
        next_value = values[i]

    returns = torch.cat(returns).detach()
    values = torch.cat(values)
    log_probs = torch.cat(log_probs)
    actions = torch.stack(actions)
    states = torch.cat(states)

    # PPO update
    for _ in range(Config.EPOCHS):
        logits, new_values = policy(states)
        new_dists = torch.distributions.Categorical(logits=logits)
        new_log_probs = new_dists.log_prob(actions)

        ratio = (new_log_probs - log_probs.detach()).exp()
        advantage = (returns - values).detach()
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - Config.CLIP_EPS, 1.0 + Config.CLIP_EPS) * advantage
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = Config.VALUE_COEF * (returns - new_values).pow(2).mean()
        entropy = Config.ENTROPY_COEF * new_dists.entropy().mean()

        loss = policy_loss + value_loss - entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Logging
    episode_rewards.append(total_reward)
    coverage_progress.append(np.mean(coverage_vals))

    print(f"Update {update + 1}/{NUM_UPDATES} | Reward: {total_reward:.2f} | Coverage: {coverage_progress[-1]:.2f}")

# Save model
torch.save(policy.state_dict(), "ppo_uav_model.pt")

# Plot training metrics
def plot_metrics(rewards, coverage):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title('Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    plt.subplot(1, 2, 2)
    plt.plot(coverage)
    plt.title('Mean Coverage During Episode')
    plt.xlabel('Episode')
    plt.ylabel('Coverage')

    plt.tight_layout()
    plt.savefig("training_metrics.png")
    plt.show()

plot_metrics(episode_rewards, coverage_progress)
