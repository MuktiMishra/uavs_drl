import torch
import yaml
import time
from uav_env import UAVEnv
from ppo_cnn_model import CNNActorCritic

# === Load Config ===
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

env = UAVEnv(config)
state = env.reset()

# === Load Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNActorCritic(input_channels=7, num_actions=8).to(device)
model.load_state_dict(torch.load("ppo_uav_model.pth", map_location=device))
model.eval()

# === Run One Episode ===
done = False
total_reward = 0
while not done:
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        action, _, _ = model.act(state_tensor)
    state, reward, done, info = env.step(action)
    total_reward += reward

    env.render()
    time.sleep(0.5)  # Delay for visualization

print("Simulation Complete. Total Reward:", total_reward)
