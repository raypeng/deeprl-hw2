import numpy as np
import random


class Sample:
    def __init__(self, _state, _action, _reward, _next_state, _is_terminal):
        self.state = _state
        self.action = _action
        self.reward = _reward
        self.next_state = _next_state
        self.is_terminal = _is_terminal


class ReplayMemory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buf = [None for _ in range(self.max_size)]
        self.curr_size = 0
        self.curr_index = 0

    def __iter__(self):
        for x in self._buf:
            yield x

    def __getitem__(self, i):
        # assert i < self.curr_size, 'ReplayMemory index {0} out of bound {1}'.format(i, self.curr_size)
        return self._buf[i]

    def __len__(self):
        return self.curr_size

    @property
    def _buf(self):
        if self.curr_size < self.max_size:
            buf = self.buf[:self.curr_size]
        else:
            buf = self.buf[self.curr_index:] + self.buf[:self.curr_index-1]
        return buf

    def append(self, state, action, reward, next_state, is_terminal):
        s = Sample(state.astype(np.uint8), action, reward, next_state.astype(np.uint8), is_terminal)
        self.buf[self.curr_index] = s
        self.curr_index = (self.curr_index + 1) % self.max_size
        self.curr_size = min(self.curr_size + 1, self.max_size)

    def sample(self, batch_size):
        # assert batch_size <= self.curr_size
        return random.sample(self._buf, batch_size)

    def clear(self):
        self.buf = [None for _ in range(self.max_size)]
        self.curr_index = 0
        self.curr_size = 0
