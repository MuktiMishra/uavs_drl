import torch
import numpy as np
import pandas as pd
import os
from uav_env import SimpleUAVEnv, CNNActorCritic  # Reuse model/env from train.py

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load environment and model
env = SimpleUAVEnv()
model = CNNActorCritic().to(device)
model.load_state_dict(torch.load("model_output/uav_ppo_cnn_model.pth", map_location=device))
model.eval()

# Simulation parameters
num_episodes = 10
max_steps = 50
simulation_records = []

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    episode_data = []

    for step in range(max_steps):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, _ = model(state_tensor)
            probs = torch.softmax(logits, dim=1)
            action = torch.argmax(probs, dim=1).item()

        next_state, reward, done, info = env.step(action)

        episode_data.append({
            'episode': episode,
            'step': step,
            'action': action,
            'reward': reward,
            'energy_used': info['energy_used'],
            'mission_time': info['mission_time'],
            'risk_score': info['risk_score'],
            'battery': info['battery'],
            'time_left': info['time_left'],
            'f2_energy': info['f2_energy'],
            'f3_time': info['f3_time'],
            'f4_risk': info['f4_risk']
        })

        total_reward += reward
        state = next_state

        if done:
            break

    print(f"Episode {episode + 1}/{num_episodes} - Total Reward: {total_reward:.4f}")
    simulation_records.extend(episode_data)

# Save simulation results
os.makedirs("model_output", exist_ok=True)
pd.DataFrame(simulation_records).to_csv("model_output/simulation_results.csv", index=False)
