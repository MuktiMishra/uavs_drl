# import pygame
# import matplotlib.pyplot as plt  # for plotting graphs
# from uav_environment import UAVEnvironment 

# print(dir(UAVEnvironment))  
# env = UAVEnvironment()
# print(hasattr(env, 'reset'))  
# pygame.init()

# # Grid dimensions
# CELL_SIZE = 40
# GRID_H, GRID_W = 15, 15

# # Add extra height to display battery info bar
# INFO_BAR_HEIGHT = 40  
# screen = pygame.display.set_mode((GRID_W * CELL_SIZE, GRID_H * CELL_SIZE + INFO_BAR_HEIGHT))
# pygame.display.set_caption("UAV Simulation")

# # Colors
# WHITE = (255, 255, 255)
# BLACK = (0, 0, 0)
# RED = (255, 0, 0)
# GREEN = (0, 255, 0)
# BLUE = (0, 0, 255)
# GRAY = (180, 180, 180)

# env = UAVEnvironment()
# state = env.reset()
# done = False
# step_count = 0

# font = pygame.font.SysFont(None, 24)

# # Lists to collect data for plotting
# battery_levels = []
# rewards = []
# coverage_percentages = []

# while not done and step_count < 1000:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             done = True

#     action = env.action_space.sample()
#     next_state, reward, done, info = env.step(action)

#     # Collect data at this step
#     battery_levels.append(env.battery)
#     rewards.append(reward)
#     coverage_map = env.coverage_map.numpy()
#     coverage_percentage = coverage_map.sum() / (GRID_H * GRID_W) * 100  # percentage coverage
#     coverage_percentages.append(coverage_percentage)

#     # Clear screen
#     screen.fill(WHITE)

#     # Draw grid lines and cells
#     for x in range(GRID_H):
#         for y in range(GRID_W):
#             rect = pygame.Rect(y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE)
#             pygame.draw.rect(screen, BLACK, rect, 1)  # grid cell border

#     # Draw UAV position (red)
#     x, y = env.uav_pos
#     uav_rect = pygame.Rect(y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE)
#     pygame.draw.rect(screen, RED, uav_rect)

#     # Draw coverage map (green smaller rectangle inside cell)
#     for i in range(GRID_H):
#         for j in range(GRID_W):
#             if coverage_map[i, j] == 1:
#                 cov_rect = pygame.Rect(j * CELL_SIZE + 10, i * CELL_SIZE + 10, CELL_SIZE - 20, CELL_SIZE - 20)
#                 pygame.draw.rect(screen, GREEN, cov_rect)

#     # Draw battery bar background (gray)
#     battery_bar_x = 0
#     battery_bar_y = GRID_H * CELL_SIZE + 10
#     battery_bar_width = GRID_W * CELL_SIZE
#     battery_bar_height = 20

#     pygame.draw.rect(screen, GRAY, (battery_bar_x, battery_bar_y, battery_bar_width, battery_bar_height))

#     # Calculate battery fill length proportional to battery left
#     fill_length = int(battery_bar_width * (env.battery / env.init_battery))
#     pygame.draw.rect(screen, GREEN, (battery_bar_x, battery_bar_y, fill_length, battery_bar_height))

#     # Display battery percentage text on the bar
#     battery_percent = int((env.battery / env.init_battery) * 100)
#     battery_text = font.render(f"Battery: {battery_percent}%", True, BLACK)
#     screen.blit(battery_text, (10, battery_bar_y - 25))

#     pygame.display.flip()

#     pygame.time.delay(200)  # slow down steps to see movement

#     step_count += 1

# pygame.quit()
# print("Simulation ended.")


# steps = list(range(step_count))

# plt.figure(figsize=(15, 5))

# #battery  => saved battery leval to a np array
# plt.subplot(1, 3, 1)
# plt.plot(steps, battery_levels, color='green')
# plt.title("Battery Level Over Time")
# plt.xlabel("Step")
# plt.ylabel("Battery Level")

# #reward graph => saved reward 
# plt.subplot(1, 3, 2)
# plt.plot(steps, rewards, color='blue')
# plt.title("Reward Over Time")
# plt.xlabel("Step")
# plt.ylabel("Reward")

# #coverage graph => total coverage per step
# plt.subplot(1, 3, 3)
# plt.plot(steps, coverage_percentages, color='orange')
# plt.title("Coverage Percentage Over Time")
# plt.xlabel("Step")
# plt.ylabel("Coverage (%)")

# plt.tight_layout()
# plt.show()
import pygame
import torch
import matplotlib.pyplot as plt
from uav_environment import UAVEnvironment
from ppo_training import CNNActorCritic  # assuming you saved the PPO model code as cnn_ppo_model.py

# Initialize Pygame
grid_H, grid_W = 15, 15
CELL_SIZE = 40
INFO_BAR_HEIGHT = 40
screen = pygame.display.set_mode((grid_W * CELL_SIZE, grid_H * CELL_SIZE + INFO_BAR_HEIGHT))
pygame.display.set_caption("UAV PPO Simulation")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
GRAY = (180, 180, 180)
font = pygame.font.SysFont(None, 24)

# Load environment and trained PPO agent
env = UAVEnvironment()
model = CNNActorCritic(input_shape=(7, 15, 15), n_actions=4)
model.load_state_dict(torch.load("ppo_uav_model.pth"))
model.eval()

state = env.reset()
done = False
step_count = 0
battery_levels = []
rewards = []
coverage_percentages = []

while not done and step_count < 1000:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_probs, _ = model(state_tensor)
        action = torch.argmax(action_probs, dim=1).item()

    state, reward, done, _ = env.step(action)

    battery_levels.append(env.battery)
    rewards.append(reward)
    coverage_map = env.coverage_map.numpy()
    coverage_percentage = coverage_map.sum() / (grid_H * grid_W) * 100
    coverage_percentages.append(coverage_percentage)

    screen.fill(WHITE)
    for x in range(grid_H):
        for y in range(grid_W):
            rect = pygame.Rect(y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, BLACK, rect, 1)

    x, y = env.uav_pos
    uav_rect = pygame.Rect(y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, RED, uav_rect)

    for i in range(grid_H):
        for j in range(grid_W):
            if coverage_map[i, j] == 1:
                cov_rect = pygame.Rect(j * CELL_SIZE + 10, i * CELL_SIZE + 10, CELL_SIZE - 20, CELL_SIZE - 20)
                pygame.draw.rect(screen, GREEN, cov_rect)

    battery_bar_x = 0
    battery_bar_y = grid_H * CELL_SIZE + 10
    battery_bar_width = grid_W * CELL_SIZE
    battery_bar_height = 20
    pygame.draw.rect(screen, GRAY, (battery_bar_x, battery_bar_y, battery_bar_width, battery_bar_height))
    fill_length = int(battery_bar_width * (env.battery / env.init_battery))
    pygame.draw.rect(screen, GREEN, (battery_bar_x, battery_bar_y, fill_length, battery_bar_height))
    battery_percent = int((env.battery / env.init_battery) * 100)
    battery_text = font.render(f"Battery: {battery_percent}%", True, BLACK)
    screen.blit(battery_text, (10, battery_bar_y - 25))

    pygame.display.flip()
    pygame.time.delay(200)
    step_count += 1

pygame.quit()
print("Simulation ended.")

steps = list(range(step_count))
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(steps, battery_levels, color='green')
plt.title("Battery Level Over Time")
plt.xlabel("Step")
plt.ylabel("Battery Level")
plt.subplot(1, 3, 2)
plt.plot(steps, rewards, color='blue')
plt.title("Reward Over Time")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.subplot(1, 3, 3)
plt.plot(steps, coverage_percentages, color='orange')
plt.title("Coverage Percentage Over Time")
plt.xlabel("Step")
plt.ylabel("Coverage (%)")
plt.tight_layout()
plt.show()

