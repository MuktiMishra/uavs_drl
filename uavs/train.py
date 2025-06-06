import torch
import numpy as np
from environment import MultiUAVEnv
from agent import DQNAgent
import yaml

def main():
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    env_cfg = config["environment"]
    agent_cfg = config["agent"]
    train_cfg = config["training"]

    env = MultiUAVEnv(config_path="config.yaml", render_mode="human" if train_cfg["render"] else None)
    num_agents = env_cfg["num_agents"]

    # Create one DQN agent per UAV
    agents = [DQNAgent(agent_cfg["state_dim"], agent_cfg["action_dim"], agent_cfg) for _ in range(num_agents)]

    episodes = train_cfg["episodes"]
    max_steps = train_cfg["max_steps_per_episode"]

    for ep in range(episodes):
        states = env.reset()
        total_rewards = [0.0 for _ in range(num_agents)]

        for step in range(max_steps):
            actions = []
            for i, agent in enumerate(agents):
                actions.append(agent.act(states[i]))

            next_states, rewards, done, truncated, _ = env.step(actions)
            for i, agent in enumerate(agents):
                agent.remember(states[i], actions[i], rewards[i], next_states[i], done)
                agent.learn()
                total_rewards[i] += rewards[i]

            states = next_states
            if done:
                break

        print(f"Episode {ep+1}/{episodes} - Rewards: {total_rewards} - Epsilon: {agents[0].epsilon:.3f}")

    env.close()


if __name__ == "__main__":
    main()
