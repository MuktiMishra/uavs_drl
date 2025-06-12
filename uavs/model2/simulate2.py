# import pygame
# import torch
# import time
# from uav_env import CNNActorCritic, SimpleUAVEnv  # make sure these match your actual filenames

# # Initialize constants
# CELL_SIZE = 40
# GRID_SIZE = 15
# WINDOW_SIZE = GRID_SIZE * CELL_SIZE

# # Colors
# WHITE = (255, 255, 255)
# GRAY = (180, 180, 180)
# BLACK = (0, 0, 0)
# RED = (255, 0, 0)
# GREEN = (0, 255, 0)
# BLUE = (0, 0, 255)
# YELLOW = (255, 255, 0)
# PURPLE = (160, 32, 240)

# # Load model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = CNNActorCritic().to(device)
# model.load_state_dict(torch.load("model_output/uav_ppo_cnn_model.pth", map_location=device))
# model.eval()

# # Initialize environment
# env = SimpleUAVEnv()
# state = env.reset()

# # Pygame setup
# pygame.init()
# screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
# pygame.display.set_caption("UAV PPO Simulation")

# def draw_grid(env):
#     screen.fill(WHITE)
#     for y in range(GRID_SIZE):
#         for x in range(GRID_SIZE):
#             rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)

#             # Grey cell for already visited
#             if env.coverage_map[y, x] == 1:
#                 pygame.draw.rect(screen, GRAY, rect)

#             # Draw victims in RED
#             if env.victim_map[y, x] == 1:
#                 pygame.draw.rect(screen, RED, rect)

#             # Draw obstacles in BLACK
#             if env.obstacle_map[y, x] == 1:
#                 pygame.draw.rect(screen, BLACK, rect)

#             # UAV in GREEN
#             if env.uav_pos == (y, x):
#                 pygame.draw.rect(screen, GREEN, rect)

#             pygame.draw.rect(screen, BLACK, rect, 1)

#     pygame.display.flip()

# def select_action(model, state):
#     state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
#     logits, _ = model(state_tensor)
#     probs = torch.softmax(logits, dim=1)
#     action = torch.argmax(probs, dim=1).item()
#     return action

# # Simulation loop
# done = False
# step = 0
# while not done and step < env.max_steps:
#     draw_grid(env)
#     time.sleep(0.3)

#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             done = True

#     action = select_action(model, state)
#     state, reward, done, info = env.step(action)
#     step += 1
#     print(f"Step {step}: Action={action}, Reward={reward:.4f}, Pos={env.uav_pos}")

# pygame.quit()
import pygame
import torch
import time
from uav_env import CNNActorCritic, SimpleUAVEnv  # Ensure correct import

# Constants
CELL_SIZE = 40
GRID_SIZE = 15
WINDOW_SIZE = GRID_SIZE * CELL_SIZE

# Colors
WHITE = (255, 255, 255)
GRAY = (180, 180, 180)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNActorCritic().to(device)
model.load_state_dict(torch.load("model_output/uav_ppo_cnn_model.pth", map_location=device))
model.eval()

# Initialize environment
env = SimpleUAVEnv()
state = env.reset()

# PyGame setup
pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("UAV PPO Simulation")

def draw_grid(env):
    screen.fill(WHITE)
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)

            # If visited, draw grey
            if env.coverage_map[y, x] == 1:
                pygame.draw.rect(screen, GRAY, rect)

            # If victim not yet visited, draw red
            if env.victim_map[y, x] == 1 and env.coverage_map[y, x] == 0:
                pygame.draw.rect(screen, RED, rect)

            # Obstacles are always black
            if env.obstacle_map[y, x] == 1:
                pygame.draw.rect(screen, BLACK, rect)

            # UAV position
            if env.uav_pos == (y, x):
                pygame.draw.rect(screen, GREEN, rect)

            pygame.draw.rect(screen, BLACK, rect, 1)  # Cell border

    pygame.display.flip()

def select_action(model, state):
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    logits, _ = model(state_tensor)
    probs = torch.softmax(logits, dim=1)
    action = torch.argmax(probs, dim=1).item()
    return action

# Simulation loop
done = False
step = 0
max_steps = 200  # Increased number of steps

while not done and step < max_steps:
    draw_grid(env)
    time.sleep(0.3)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    action = select_action(model, state)
    state, reward, done, info = env.step(action)

    step += 1
    print(f"Step {step}: Action={action}, Reward={reward:.4f}, Pos={env.uav_pos}")

pygame.quit()

