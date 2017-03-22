import gym
import utils

import numpy as np
import random


class AtariEnv:
    def __init__(self, env_name, state_dim=84, n_action_repeat=4):
        self.env = gym.make(env_name)
        self.state_dim = state_dim
        self.n_action_repeat = n_action_repeat
        self.action_size = self.env.action_space.n

    def random_action(self):
        return self.env.action_space.sample()

    def new_game(self):
        screen = self.env.reset()
        self.lives = self.env.ale.lives()
        is_terminal = False
        reward = 0
        states = [utils.preprocess_frame(screen) for _ in range(self.n_action_repeat)]
        return self._dstack(states), reward, is_terminal

    def step(self, action):
        states = []
        for _ in range(self.n_action_repeat):
            screen, reward, is_terminal, _ = self.env.step(action)
            current_lives = self.env.ale.lives()
            is_terminal = self.lives > current_lives
            states.append(utils.preprocess_frame(screen))
            if is_terminal:
                break
        if not is_terminal:
            self.lives = current_lives
        reward = max(-1, min(1, reward))
        return self._dstack(states), reward, is_terminal

    def _dstack(self, states):
        if len(states) < self.n_action_repeat:
            states += [states[-1] for _ in range(self.n_action_repeat - len(states))]
        assert len(states) == self.n_action_repeat
        state = states[0]
        for s in states[1:]:
            state = np.dstack((state, s))
        assert state.shape == (self.state_dim, self.state_dim, self.n_action_repeat), state.shape
        return state
