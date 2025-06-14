# config.py

GRID_SIZE = 10
NUM_UAVS = 2
WIND_COVERAGE = 0.1  # 10% of the grid

# Wind configuration
WIND_DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
DIRECTION_VECTORS = {
    "N": (-1, 0), "NE": (-1, 1), "E": (0, 1), "SE": (1, 1),
    "S": (1, 0), "SW": (1, -1), "W": (0, -1), "NW": (-1, -1)
}
UAV_DIRECTIONS = list(DIRECTION_VECTORS.values())
WIND_EFFECT_PROB_THRESHOLD = 0.6
MAX_SPEED_WITH_WIND = 3
MIN_SPEED_AGAINST_WIND = 0

OBSTACLE_COUNT = 5

# Rendering Colors
COLOR_BG = (255, 255, 255)
COLOR_UAV = [(0, 0, 255), (0, 255, 0)]
COLOR_WIND = (135, 206, 250)
COLOR_OBSTACLE = (0, 0, 0)
COLOR_VISITED = (200, 200, 200)

# Training parameters
EPISODES = 1000
LEARNING_RATE = 0.01
DISCOUNT_FACTOR = 0.99
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.1
MAX_EPSILON = 1.0

# Q-table size
STATE_SPACE = GRID_SIZE * GRID_SIZE
ACTION_SPACE = len(UAV_DIRECTIONS)
