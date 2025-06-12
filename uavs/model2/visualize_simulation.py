# import pygame
# import torch
# import time
# from uav_env import CNNActorCritic, SimpleUAVEnv

# # Initialize constants
# CELL_SIZE = 40
# GRID_SIZE = 15
# WINDOW_SIZE = GRID_SIZE * CELL_SIZE

# # Colors
# WHITE = (255, 255, 255)
# GRAY = (180, 180, 180)
# BLACK = (0, 0, 0)
# GREEN = (0, 200, 0)
# RED = (200, 0, 0)
# BLUE = (0, 0, 255)
# YELLOW = (255, 255, 0)
# BROWN = (0, 255, 0)

# # Initialize PyGame
# pygame.init()
# win = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
# pygame.display.set_caption("UAV PPO Simulation")

# def draw_grid(env, uav_pos):
#     win.fill(WHITE)
    
#     for y in range(GRID_SIZE):
#         for x in range(GRID_SIZE):
#             rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
#             pygame.draw.rect(win, GRAY, rect, 1)  # Cell border

#             if env.coverage_map[y][x] == 1:
#                 pygame.draw.rect(win, BROWN, rect)  # Covered cells
#             if env.obstacle_map[y][x] == 1:
#                 pygame.draw.rect(win, RED, rect)
#             elif env.victim_map[y][x] == 1:
#                 pygame.draw.rect(win, BLUE, rect)

#     # Draw UAV on top
#     uav_rect = pygame.Rect(uav_pos[1] * CELL_SIZE, uav_pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
#     pygame.draw.rect(win, YELLOW, uav_rect)

#     pygame.display.update()

# def simulate_visual():
#     env = SimpleUAVEnv()
#     state = env.reset()

#     model = CNNActorCritic()
#     model.load_state_dict(torch.load("model_output/uav_ppo_cnn_model.pth"))
#     model.eval()

#     clock = pygame.time.Clock()
#     running = True
#     step = 0

#     while running:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False

#         state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
#         with torch.no_grad():
#             logits, _ = model(state_tensor)
#             probs = torch.softmax(logits, dim=1)
#             action = torch.argmax(probs, dim=1).item()

#         next_state, reward, done, info = env.step(action)
#         draw_grid(env, env.uav_pos)
#         state = next_state
#         step += 1

#         time.sleep(0.2)
#         clock.tick(10)

#         if done or step >= 50:
#             print("Simulation Ended")
#             time.sleep(1)
#             running = False

#     pygame.quit()

# if __name__ == "__main__":
#     simulate_visual()
import pygame
import torch
import time
from uav_env import CNNActorCritic, SimpleUAVEnv

# Initialize constants
CELL_SIZE = 40
GRID_SIZE = 15
WINDOW_SIZE = GRID_SIZE * CELL_SIZE

# Colors
WHITE = (255, 255, 255)
GRAY = (169, 169, 169)    # Visited cells
BLACK = (0, 0, 0)         # Obstacles
RED = (255, 0, 0)         # Victims
BLUE = (0, 0, 255)        # UAV Agent
BORDER = (0, 0, 0)  # Cell borders

# Initialize PyGame
pygame.init()
win = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("UAV PPO Simulation")

def draw_grid(env, uav_pos):
    win.fill(WHITE)
    
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)

            # Draw each grid cell based on type
            if env.obstacle_map[y][x] == 1:
                pygame.draw.rect(win, BLACK, rect)  # Obstacles
            elif env.victim_map[y][x] == 1:
                pygame.draw.rect(win, RED, rect)    # Victims
            elif env.coverage_map[y][x] == 1:
                pygame.draw.rect(win, GRAY, rect)   # Visited cells

            pygame.draw.rect(win, BORDER, rect, 1)  # Cell border

    # Draw UAV on top
    uav_rect = pygame.Rect(uav_pos[1] * CELL_SIZE, uav_pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(win, BLUE, uav_rect)

    pygame.display.update()

def simulate_visual():
    env = SimpleUAVEnv()
    state = env.reset()

    model = CNNActorCritic()
    model.load_state_dict(torch.load("model_output/uav_ppo_cnn_model.pth"))
    model.eval()

    clock = pygame.time.Clock()
    running = True
    step = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(state_tensor)
            probs = torch.softmax(logits, dim=1)
            action = torch.argmax(probs, dim=1).item()

        next_state, reward, done, info = env.step(action)
        draw_grid(env, env.uav_pos)
        state = next_state
        step += 1

        time.sleep(0.2)
        clock.tick(10)

        if done or step >= 50:
            print("Simulation Ended")
            time.sleep(1)
            running = False

    pygame.quit()

if __name__ == "__main__":
    simulate_visual()
