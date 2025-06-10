import torch
import yaml
import matplotlib.pyplot as plt
import numpy as np
from uav_env import UAVEnv
from ppo_cnn_model import CNNActorCritic

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

env_cfg = config["env_config"]
action_size = len(config["ACTIONS"])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = UAVEnv(config)
model = CNNActorCritic(env_cfg['channels'], action_size).to(device)
model.load_state_dict(torch.load("ppo_uav_model.pth", map_location=device))
model.eval()


rewards = []
coverages = []
states = []

for episode in range(10):
    state = env.reset()
    total_reward = 0
    steps = 0

    while True:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action, _, _ = model.act(state_tensor)
        state, reward, done, info = env.step(action)

        total_reward += reward
        steps += 1

        if done:
            break

    rewards.append(total_reward)
    coverages.append(info.get("coverage", 0))
    print(f"Episode {episode} - Total Reward: {total_reward:.2f}, Coverage: {info.get('coverage', 0):.2f}")


plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(rewards)
plt.title("Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")

plt.subplot(1, 2, 2)
plt.plot(coverages)
plt.title("Coverage per Episode")
plt.xlabel("Episode")
plt.ylabel("Coverage (%)")

plt.tight_layout()
plt.savefig("evaluation_plots.png")
plt.show()
