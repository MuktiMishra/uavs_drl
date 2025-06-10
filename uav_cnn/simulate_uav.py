import pygame
import torch
import matplotlib.pyplot as plt
from uav_environment import UAVEnvironment
from ppo_training import CNNActorCritic

pygame.init()

# Constants
grid_H, grid_W = 15, 15
CELL_SIZE = 40
INFO_BAR_HEIGHT = 40

screen = pygame.display.set_mode((grid_W * CELL_SIZE, grid_H * CELL_SIZE + INFO_BAR_HEIGHT))
pygame.display.set_caption("UAV PPO Simulation")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 100, 255)
YELLOW = (255, 255, 0)
GRAY = (180, 180, 180)
font = pygame.font.SysFont(None, 24)

# Load environment and model
env = UAVEnvironment()
model = CNNActorCritic(input_channels=7, num_actions=4)
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
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            done = True

    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_logits, _ = model(state_tensor)
        action = torch.argmax(action_logits, dim=1).item()

    state, reward, done, _ = env.step(action)

    battery_levels.append(env.battery)
    rewards.append(reward)
    coverage_percentage = env.coverage_map.sum().item() / (grid_H * grid_W) * 100
    coverage_percentages.append(coverage_percentage)

    # === Drawing ===
    screen.fill(WHITE)

    # Grid
    for x in range(grid_H):
        for y in range(grid_W):
            rect = pygame.Rect(y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, BLACK, rect, 1)

    # Coverage
    for i in range(grid_H):
        for j in range(grid_W):
            if env.coverage_map[i, j] == 1:
                rect = pygame.Rect(j * CELL_SIZE + 10, i * CELL_SIZE + 10, CELL_SIZE - 20, CELL_SIZE - 20)
                pygame.draw.rect(screen, GREEN, rect)

    # Victims
    for vx, vy in env.victims:
        victim_color = YELLOW if (vx, vy) in env.reached_victims else BLUE
        rect = pygame.Rect(vy * CELL_SIZE + 8, vx * CELL_SIZE + 8, CELL_SIZE - 16, CELL_SIZE - 16)
        pygame.draw.rect(screen, victim_color, rect)

    # UAV
    ux, uy = env.uav_pos
    uav_rect = pygame.Rect(uy * CELL_SIZE, ux * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, RED, uav_rect)

    # Battery bar
    bar_y = grid_H * CELL_SIZE + 10
    full_width = grid_W * CELL_SIZE
    pygame.draw.rect(screen, GRAY, (0, bar_y, full_width, 20))
    filled = int(full_width * (env.battery / env.init_battery))
    pygame.draw.rect(screen, GREEN, (0, bar_y, filled, 20))
    battery_text = font.render(f"Battery: {int(env.battery / env.init_battery * 100)}%", True, BLACK)
    screen.blit(battery_text, (10, bar_y - 25))

    pygame.display.flip()
    pygame.time.delay(200)
    step_count += 1

pygame.quit()
print("Simulation ended.")

# Plot statistics
input("Press Enter to show plots...")

steps = list(range(step_count))
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(steps, battery_levels, color='green')
plt.title("Battery Level Over Time")
plt.xlabel("Step")
plt.ylabel("Battery")

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
