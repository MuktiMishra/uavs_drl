import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class CNNActorCritic(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(CNNActorCritic, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)

        self.flatten_size = 8 * 15 * 15  # 8 channels after conv, 15x15 grid

        self.fc_actor = nn.Linear(self.flatten_size, num_actions)
        self.fc_critic = nn.Linear(self.flatten_size, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        action_logits = self.fc_actor(x)
        state_values = self.fc_critic(x)
        return action_logits, state_values

    def act(self, state):
        action_logits, _ = self.forward(state)
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()

    def evaluate(self, state, action):
        action_logits, state_values = self.forward(state)
        dist = Categorical(logits=action_logits)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, state_values.squeeze(), dist_entropy
