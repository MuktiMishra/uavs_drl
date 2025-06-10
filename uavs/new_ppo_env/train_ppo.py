import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import yaml
import os
import csv
from uav_env import UAVEnv
from ppo_cnn_model import CNNActorCritic

# === Load config ===
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

env_cfg = config["env_config"]
ppo_cfg = config["ppo_config"]
action_size = len(config["ACTIONS"])

# === Init ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = UAVEnv(config)
model = CNNActorCritic(env_cfg['channels'], action_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=ppo_cfg['learning_rate'])

# === PPO Storage ===
class RolloutBuffer:
    def __init__(self):
        self.states, self.actions, self.logprobs, self.rewards, self.is_terminals = [], [], [], [], []

    def clear(self):
        self.__init__()

buffer = RolloutBuffer()

def compute_returns(rewards, gamma, dones):
    R = 0
    returns = []
    for reward, done in zip(reversed(rewards), reversed(dones)):
        R = reward + gamma * R * (1 - done)
        returns.insert(0, R)
    return returns

# === Training Loop ===
timestep = 0
reward_log = []
coverage_log = []

for update in range(int(ppo_cfg['total_timesteps'] / ppo_cfg['update_timestep'])):
    state = env.reset()
    ep_rewards, ep_coverage = 0, 0

    for t in range(ppo_cfg['update_timestep']):
        timestep += 1

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        action, logprob, _ = model.act(state_tensor)

        next_state, reward, done, info = env.step(action)

        buffer.states.append(state_tensor)
        buffer.actions.append(torch.tensor(action).to(device))
        buffer.logprobs.append(logprob.detach())
        buffer.rewards.append(reward)
        buffer.is_terminals.append(done)

        state = next_state
        ep_rewards += reward

        if done:
            break

    # PPO Update
    returns = compute_returns(buffer.rewards, ppo_cfg['gamma'], buffer.is_terminals)
    returns = torch.tensor(returns).to(device).detach()
    old_states = torch.cat(buffer.states).to(device).detach()
    old_actions = torch.stack(buffer.actions).to(device).detach()
    old_logprobs = torch.stack(buffer.logprobs).to(device).detach()

    for _ in range(ppo_cfg['k_epochs']):
        logprobs, state_values, dist_entropy = model.evaluate(old_states, old_actions)
        advantages = returns - state_values.detach()

        ratio = torch.exp(logprobs - old_logprobs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - ppo_cfg['eps_clip'], 1 + ppo_cfg['eps_clip']) * advantages

        loss = -torch.min(surr1, surr2) + \
               ppo_cfg['value_loss_coef'] * (state_values - returns).pow(2) - \
               ppo_cfg['entropy_coef'] * dist_entropy

        optimizer.zero_grad()
        loss.mean().backward()
        nn.utils.clip_grad_norm_(model.parameters(), ppo_cfg['max_grad_norm'])
        optimizer.step()

    reward_log.append(ep_rewards)
    coverage_log.append(info.get("coverage", 0))
    buffer.clear()

    print(f"Update {update}, Reward: {ep_rewards:.2f}, Coverage: {info.get('coverage', 0):.2f}")

# Save results
np.savetxt("reward_log.csv", reward_log, delimiter=",")
np.savetxt("coverage_log.csv", coverage_log, delimiter=",")
torch.save(model.state_dict(), "ppo_uav_model.pth")
