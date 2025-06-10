class Config:
    # Environment
    GRID_H = 15
    GRID_W = 15
    CHANNELS = 7  # per description

    # CNN model
    CONV_FILTERS = 8
    KERNEL_SIZE = 3

    # PPO Hyperparams
    GAMMA = 0.99
    LAMBDA = 0.95
    CLIP_EPS = 0.2
    LR = 3e-4
    VALUE_COEF = 0.5
    ENTROPY_COEF = 0.01
    BATCH_SIZE = 64
    EPOCHS = 4
    TOTAL_TIMESTEPS = 100000
    UPDATE_TIMESTEP = 2048

    # Action space size
    NUM_ACTIONS = 4

    # Battery & Energy model
    BASE_ENERGY = 0.005
    # ENERGY_ACTION_COSTS = [0.0, 0.005, 0.01, 0.015]  # per action
    ENERGY_ACTION_COSTS = [0.1, 0.1, 0.1, 0.1, 0.2, 0.05]  # example costs for actions 0-5

    TASK_ENERGY = 0.005
    ENERGY_NOISE_STD = 0.001

    # Wind
    WIND_K = 0.03
