import gym
from gym import wrappers
import utils

import os
import random
import numpy as np
from datetime import datetime
from collections import deque


class AtariEnv:
    def __init__(self, env_name, model, do_render=False, make_video=False, video_dir=None, state_dim=84, n_frame_input=4):
        self.env = gym.make(env_name)
        if make_video:
            if video_dir:
                monitor_path = video_dir
                use_force = True
            else:
                dt = datetime.now().strftime('%m-%d-%H:%M:%S')
                monitor_path = os.path.join('monitor', model, dt)
                use_force = False
            every_iter = lambda x: True
            self.env = wrappers.Monitor(self.env,
                                        monitor_path,
                                        force=use_force,
                                        video_callable=every_iter,
                                        write_upon_reset=True)
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

    def step(self, action, include_noclip=False):
        screen, reward, is_terminal, _ = self.env.step(action)
        if self.do_render:
            self.env.render()
        self.history.append(utils.preprocess_frame(screen))
        self.history.popleft()
        clipped_reward = max(-1, min(1, reward))
        if include_noclip:
            return self._dstack(self.history), clipped_reward, is_terminal, reward
        else:
            return self._dstack(self.history), clipped_reward, is_terminal

    def _dstack(self, states):
        # assert len(states) == self.n_frame_input
        dim = states[0].shape[0]
        state = np.array([], dtype=np.float32).reshape(dim, dim, 0)
        for s in states:
            state = np.dstack((state, s))
        # assert state.shape == (self.state_dim, self.state_dim, self.n_frame_input), state.shape
        return state
