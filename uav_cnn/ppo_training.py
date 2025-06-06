import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class CNNActorCritic(nn.Module):
    def __init__(self, input_channels=7, num_actions=4):
        super(CNNActorCritic, self).__init__()

        # CNN layers to extract spatial features from (7, 15, 15) input
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Compute the flattened feature size after conv + pool layers
        # Input: (64, 15, 15) -> Pool: (64, 7, 7)
        self.flattened_size = 64 * 7 * 7

        # Shared dense layer after flattening
        self.fc_shared = nn.Linear(self.flattened_size, 256)

        # Actor head (outputs action probabilities)
        self.actor = nn.Linear(256, num_actions)

        # Critic head (outputs state value estimate)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)  # ensure correct dtype
        if len(x.shape) == 3:
            x = x.unsqueeze(0)  # Add batch dimension if missing

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc_shared(x))

        action_logits = self.actor(x)
        state_value = self.critic(x)

        return action_logits, state_value

    def act(self, x):
        logits, _ = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()

    def evaluate(self, x, actions):
        logits, values = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)

        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        return action_logprobs, torch.squeeze(values), dist_entropy
