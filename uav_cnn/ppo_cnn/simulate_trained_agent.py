import torch
import matplotlib.pyplot as plt
import numpy as np
import time

from sar_env import SARGridEnv
from ppo_model import CNNActorCritic

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "ppo_uav_model.pt"

# Load trained model
env = SARGridEnv()
policy = CNNActorCritic().to(DEVICE)

policy.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
policy.eval()

# Visualization setup - light theme colors
def render_grid(env, step):
    grid = np.ones((env.grid_h, env.grid_w, 3), dtype=np.uint8) * 245  # Light off-white background

    # Obstacles: soft gray
    grid[env.obstacle_map == 1] = [200, 200, 200]

    # Victims: soft coral (light warm red)
    grid[env.victim_map == 1] = [240, 128, 128]

    # Covered cells: pale turquoise
    grid[env.coverage_map == 1] = [175, 238, 238]

    # UAV position: dark cyan
    x, y = env.uav_pos
    grid[x, y] = [0, 139, 139]

    plt.imshow(grid)
    plt.title(f"Step {step} | Battery: {env.battery:.2f} | Victims left: {env.victim_map.sum()}", color='#333333')
    plt.axis('off')
    plt.pause(0.2)
    plt.clf()

# Run simulation
obs = env.reset()
done = False
step = 0

plt.figure(figsize=(6, 6))
while not done:
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits, _ = policy(obs_tensor)
    action = torch.argmax(logits, dim=-1).item()

    obs, _, done, _ = env.step(action)
    render_grid(env, step)
    step += 1

plt.close()
