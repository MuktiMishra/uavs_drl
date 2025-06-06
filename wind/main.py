import pygame
import time
import random
from env import UAVEnvironment
from config import *

pygame.init()
screen_size = GRID_SIZE * 50
screen = pygame.display.set_mode((screen_size, screen_size))
pygame.display.set_caption("UAV Wind Navigation Simulation")
clock = pygame.time.Clock()

env = UAVEnvironment()

# Q-learning setup
q_table = {}
epsilon = MAX_EPSILON

def get_q(state_key, action):
    return q_table.get((state_key, action), 0.0)

def choose_action(state_key):
    if random.random() < epsilon:
        return [random.randint(0, len(UAV_DIRECTIONS) - 1) for _ in range(NUM_UAVS)]
    else:
        best = []
        for a1 in range(len(UAV_DIRECTIONS)):
            for a2 in range(len(UAV_DIRECTIONS)):
                action = (a1, a2)
                q_value = get_q(state_key, action)
                best.append((q_value, action))
        best.sort(reverse=True)
        return list(best[0][1])

for episode in range(10):
    state = env.reset()
    state_key = tuple(env.uav_positions)
    done = False
    total_reward = 0
    step_count = 0

    while not done and step_count < 500:
        step_count += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        action = choose_action(state_key)
        next_state, reward, done = env.step(action)
        next_state_key = tuple(env.uav_positions)

        q_current = get_q(state_key, tuple(action))
        q_next = max(get_q(next_state_key, (a1, a2)) for a1 in range(len(UAV_DIRECTIONS)) for a2 in range(len(UAV_DIRECTIONS)))
        new_q = q_current + LEARNING_RATE * (reward + DISCOUNT_FACTOR * q_next - q_current)
        q_table[(state_key, tuple(action))] = new_q

        state_key = next_state_key
        total_reward += reward

        # Visualize environment
        screen.fill(COLOR_BG)
        env.render(screen, pygame)
        pygame.display.flip()
        clock.tick(10)  # Limit to 10 FPS

        print(f"Episode {episode+1} Step {step_count} Coverage: {env.covered}/{GRID_SIZE * GRID_SIZE - len(env.obstacles)}")

    print(f"Episode {episode+1} ended. Final Coverage: {env.covered}/{GRID_SIZE * GRID_SIZE - len(env.obstacles)} Total Reward: {total_reward}")
    epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

pygame.quit()
