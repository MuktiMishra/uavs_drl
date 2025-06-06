import numpy as np
import random
from config import *

class UAVEnvironment:
    def __init__(self):
        self.grid_size = GRID_SIZE
        self.num_uavs = NUM_UAVS
        self.wind_cells = {}
        self.obstacles = set()

        self._generate_obstacles()
        self._generate_wind_cells()  # generate wind once, fixed for all episodes

        self.reset()

    def reset(self):
        self.grid = np.zeros((self.grid_size, self.grid_size))
        self.visited = [[False] * self.grid_size for _ in range(self.grid_size)]
        self.covered = 0
        self.uav_positions = [(0, 0), (self.grid_size - 1, self.grid_size - 1)]
        for x, y in self.uav_positions:
            self.visited[x][y] = True
            self.covered += 1
        return self._get_state()

    def _generate_obstacles(self):
        self.obstacles.clear()
        for _ in range(OBSTACLE_COUNT):
            while True:
                x = random.randint(0, self.grid_size - 1)
                y = random.randint(0, self.grid_size - 1)
                if (x, y) not in self.obstacles and (x, y) not in [(0,0), (self.grid_size - 1, self.grid_size - 1)]:
                    self.obstacles.add((x, y))
                    break

    def _generate_wind_cells(self):
        self.wind_cells.clear()
        wind_count = int(WIND_COVERAGE * self.grid_size * self.grid_size)
        for _ in range(wind_count):
            while True:
                x = random.randint(0, self.grid_size - 2)
                y = random.randint(0, self.grid_size - 2)
                # Ensure wind cells do not overlap with obstacles
                if all((x + dx, y + dy) not in self.wind_cells and (x + dx, y + dy) not in self.obstacles for dx in range(2) for dy in range(2)):
                    direction = random.choice(UAV_DIRECTIONS)
                    strength = random.choice([1, 2])
                    for dx in range(2):
                        for dy in range(2):
                            self.wind_cells[(x + dx, y + dy)] = (direction, strength)
                    break

    def _get_state(self):
        state = np.zeros((self.grid_size, self.grid_size))
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.visited[x][y]:
                    state[x][y] = 0.5
        for x, y in self.uav_positions:
            state[x][y] = 1.0
        return state

    def step(self, actions):
        new_positions = []
        reward = 0

        for i, (x, y) in enumerate(self.uav_positions):
            dx, dy = UAV_DIRECTIONS[actions[i]]

            # Apply wind effect if present
            if (x, y) in self.wind_cells:
                wind_dx, wind_dy = self.wind_cells[(x, y)][0]
                strength = self.wind_cells[(x, y)][1]
                dx += wind_dx * (strength - 1)
                dy += wind_dy * (strength - 1)

            new_x, new_y = x + dx, y + dy

            if 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size:
                if (new_x, new_y) not in self.obstacles:
                    if not self.visited[new_x][new_y]:
                        reward += 1
                        self.visited[new_x][new_y] = True
                        self.covered += 1
                    new_positions.append((new_x, new_y))
                else:
                    new_positions.append((x, y))  # Obstacle, stay
            else:
                new_positions.append((x, y))  # Out of bounds, stay

        self.uav_positions = new_positions
        done = self.covered >= (self.grid_size * self.grid_size - len(self.obstacles))
        return self._get_state(), reward, done

    def render(self, screen, pygame):
        cell_size = 50
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(y * cell_size, x * cell_size, cell_size, cell_size)
                if (x, y) in self.obstacles:
                    pygame.draw.rect(screen, COLOR_OBSTACLE, rect)
                elif self.visited[x][y]:
                    pygame.draw.rect(screen, COLOR_VISITED, rect)
                else:
                    pygame.draw.rect(screen, COLOR_BG, rect)
                pygame.draw.rect(screen, (0, 0, 0), rect, 1)

                # Draw wind direction
                if (x, y) in self.wind_cells:
                    wind_dx, wind_dy = self.wind_cells[(x, y)][0]
                    center = (y * cell_size + cell_size // 2, x * cell_size + cell_size // 2)
                    end = (center[0] + wind_dy * 10, center[1] + wind_dx * 10)
                    pygame.draw.line(screen, (0, 0, 255), center, end, 2)

        for i, (x, y) in enumerate(self.uav_positions):
            center = (y * cell_size + cell_size // 2, x * cell_size + cell_size // 2)
            points = [
                (center[0], center[1] - 10),
                (center[0] - 10, center[1] + 10),
                (center[0] + 10, center[1] + 10)
            ]
            pygame.draw.polygon(screen, COLOR_UAV[i], points)
