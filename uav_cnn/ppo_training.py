import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim
import numpy as np

class CNNActorCritic(nn.Module):
    def __init__(self, input_channels=7, num_actions=4):
        super(CNNActorCritic, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # After pooling: (8, 7, 7)
        self.flattened_size = 8 * 7 * 7

        self.fc_shared = nn.Linear(self.flattened_size, 256)

        self.actor = nn.Linear(256, num_actions)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)  # add batch dim

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)
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


# === Dummy UAV environment placeholder ===
class DummyEnv:
    def __init__(self):
        self.observation_shape = (7, 15, 15)
        self.num_actions = 4

    def reset(self):
        # Return initial observation (random tensor for demo)
        return np.random.rand(*self.observation_shape).astype(np.float32)

    def step(self, action):
        # Dummy next state, reward, done
        next_state = np.random.rand(*self.observation_shape).astype(np.float32)
        reward = np.random.rand()  # random reward
        done = np.random.rand() > 0.95  # 5% chance to end episode
        info = {}
        return next_state, reward, done, info


# === PPO training loop skeleton ===
def train():
    env = DummyEnv()
    model = CNNActorCritic(input_channels=7, num_actions=4)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    num_episodes = 500

    gamma = 0.99  # discount factor
    eps_clip = 0.2  # PPO clip param

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        log_probs = []
        values = []
        rewards = []
        entropies = []
        states = []
        actions = []

        while not done:
            action, log_prob, entropy = model.act(state)

            next_state, reward, done, _ = env.step(action)

            # Save for training
            log_probs.append(log_prob)
            rewards.append(reward)
            entropies.append(entropy)

            # Get value for current state
            _, value = model.forward(state)
            values.append(value.squeeze())

            states.append(state)
            actions.append(torch.tensor(action))

            state = next_state

        # Compute discounted rewards (returns)
        returns = []
        discounted_sum = 0
        for r in reversed(rewards):
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)
        returns = torch.tensor(returns)

        # Convert lists to tensors
        log_probs = torch.stack(log_probs)
        values = torch.stack(values)
        entropies = torch.stack(entropies)
        actions = torch.stack(actions)

        # Advantages
        advantages = returns - values.detach()

        # PPO Loss calculation
        # New log probs from current policy for the taken actions
        new_log_probs, new_values, dist_entropy = model.evaluate(torch.stack([torch.tensor(s) for s in states]), actions)

        ratios = torch.exp(new_log_probs - log_probs.detach())
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = F.mse_loss(new_values, returns)
        entropy_loss = -dist_entropy.mean()

        loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Episode {episode+1}/{num_episodes}, Loss: {loss.item():.4f}, Reward: {sum(rewards):.2f}")

        # Save model every 100 episodes
        if (episode + 1) % 100 == 0:
            torch.save(model.state_dict(), "ppo_uav_model.pth")
            print(f"Model saved at episode {episode+1}")

    # Save final model
    torch.save(model.state_dict(), "ppo_uav_model.pth")
    print("Training completed and model saved.")


# === Run training if script executed ===
if __name__ == "__main__":
    train()
