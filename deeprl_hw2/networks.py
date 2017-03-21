import keras


class DeepQ:
    def __init__(self, num_frames, input_size, num_actions):
        self.num_frames = num_frames
        self.input_size = input_size
        self.num_actions = num_actions
        self._construct_network()

    def _construct_network(self):
        pass
