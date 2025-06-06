import gym
from gym import spaces
import pygame
import numpy as np


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size
        self.window_size = 512

        self.observation_space = spaces.Dict({
            "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
        })

        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([1, 0]),   # Right
            1: np.array([0, 1]),   # Down
            2: np.array([-1, 0]),  # Left
            3: np.array([0, -1])   # Up
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self._agent_location = None
        self._target_location = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._agent_location = np.array([0, 0])
        self._target_location = np.array([self.np_random.integers(self.size), self.np_random.integers(self.size)])
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = np.array([self.np_random.integers(self.size), self.np_random.integers(self.size)])

        observation = self._get_obs()
        if self.render_mode == "human":
            self._render_frame()
        return observation

    def step(self, action):
        direction = self._action_to_direction[action]
        self._agent_location = np.clip(self._agent_location + direction, 0, self.size - 1)

        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, False, {}

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        pix_square_size = self.window_size // self.size

        # Draw target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )

        # Draw agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size // 3,
        )

        # Draw grid lines
        for x in range(self.size + 1):
            pygame.draw.line(canvas, 0, (0, x * pix_square_size), (self.window_size, x * pix_square_size), width=1)
            pygame.draw.line(canvas, 0, (x * pix_square_size, 0), (x * pix_square_size, self.window_size), width=1)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


# ======= Run the Environment =======
if __name__ == "__main__":
    env = GridWorldEnv(render_mode="human", size=5)
    obs = env.reset()

    done = False
    while not done:
        action = env.action_space.sample()  # Random action
        obs, reward, done, _, _ = env.step(action)

    env.close()
