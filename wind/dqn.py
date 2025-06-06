# dqn.py
import numpy as np
import random

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate, gamma, epsilon, epsilon_decay, epsilon_min):
        self.q_table = np.zeros((state_size, action_size))
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        predict = self.q_table[state][action]
        target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action] += self.lr * (target - predict)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
