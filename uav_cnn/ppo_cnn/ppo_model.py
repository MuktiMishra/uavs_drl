import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np
import importlib.util
import sys
import os

# Load .config.py dynamically
config_path = os.path.join(os.path.dirname(__file__), '.config.py')
spec = importlib.util.spec_from_file_location("config", config_path)
config_module = importlib.util.module_from_spec(spec)
sys.modules["config"] = config_module
spec.loader.exec_module(config_module)

Config = config_module.Config
class CNNActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(Config.CHANNELS, Config.CONV_FILTERS, kernel_size=Config.KERNEL_SIZE, padding=1)
        self.conv2 = nn.Conv2d(Config.CONV_FILTERS, Config.CONV_FILTERS, kernel_size=Config.KERNEL_SIZE, padding=1)

        conv_out_size = Config.CONV_FILTERS * Config.GRID_H * Config.GRID_W

        self.fc1 = nn.Linear(conv_out_size, 256)
        self.fc_policy = nn.Linear(256, Config.NUM_ACTIONS)
        self.fc_value = nn.Linear(256, 1)

    def forward(self, x):
        # x shape: (batch, channels, H, W)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))

        logits = self.fc_policy(x)
        value = self.fc_value(x).squeeze(-1)
        return logits, value
