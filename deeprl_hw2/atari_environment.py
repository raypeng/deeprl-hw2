import gym
import utils

import random


class AtariEnv:
    def __init__(self, env_name, state_dim=84, n_action_repeat=4, n_frame_input=4):
        self.env = gym.make(env_name)
        self.n_action_repeat = n_action_repeat
        self.action_size = self.env.action_space.n

    def random_action(self):
        return self.env.action_space.sample()

    def new_game(self):
        screen = self.env.reset()
        self.lives = self.env.ale.lives()
        is_terminal = False
        reward = 0
        return utils.preprocess_frame(screen), reward, is_teriminal

    def step(self, action):
        for _ in range(self.n_action_repeat):
            screen, reward, is_terminal, _ = self.env.step(action)
            current_lives = self.env.ale.lives()
            is_terminal = self.lives > current_lives
            if is_terminal:
                break
        if not is_terminal:
            self.lives = current_lives
        reward = max(-1, min(1, reward))
        return utils.preprocess_frame(screen), reward, is_terminal
