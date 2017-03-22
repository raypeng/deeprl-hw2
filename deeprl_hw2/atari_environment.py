import gym
import utils

import numpy as np
import random
from collections import deque


class AtariEnv:
    def __init__(self, env_name, do_render=False, state_dim=84, n_frame_input=4):
        self.env = gym.make(env_name)
        self.state_dim = state_dim
        self.n_frame_input = n_frame_input
        self.action_size = self.env.action_space.n
        self.history = deque()
        self.do_render = do_render

    def random_action(self):
        return self.env.action_space.sample()

    def new_game(self):
        screen = self.env.reset()
        if self.do_render:
            self.env.render()
        is_terminal = False
        reward = 0
        self.history = deque([utils.preprocess_frame(screen) for _ in range(self.n_frame_input)])
        return self._dstack(self.history), reward, is_terminal

    def step(self, action):
        screen, reward, is_terminal, _ = self.env.step(action)
        if self.do_render:
            self.env.render()
        self.history.append(utils.preprocess_frame(screen))
        self.history.popleft()
        reward = max(-1, min(1, reward))
        return self._dstack(self.history), reward, is_terminal

    def _dstack(self, states):
        assert len(states) == self.n_frame_input
        dim = states[0].shape[0]
        state = np.array([], dtype=np.float32).reshape(dim, dim, 0)
        for s in states:
            state = np.dstack((state, s))
        assert state.shape == (self.state_dim, self.state_dim, self.n_frame_input), state.shape
        return state
