"""Suggested Preprocessors."""

from copy import deepcopy
import numpy as np
from scipy.misc import imresize
from PIL import Image

from deeprl_hw2 import utils
from deeprl_hw2.core import Preprocessor


class HistoryPreprocessor(Preprocessor):
    """Keeps the last k states.

    Useful for domains where you need velocities, but the state
    contains only positions.

    When the environment starts, this will just fill the initial
    sequence values with zeros k times.

    Parameters
    ----------
    history_length: int
      Number of previous states to prepend to state being processed.

    """
    def __init__(self, history_length=1):
        self.history_length = history_length
        self.buf = []

    def process_state_for_network(self, state):
        """You only want history when you're deciding the current action to take."""
        if len(self.buf) < self.history_length:
            n = self.history_length - len(self.buf)
            buf_old = deepcopy(self.buf)
            self.buf.append(state)
            return [np.zero(state.shape) for _ in range(n)] + buf_old
        else: # buf full
            buf_old = deepcopy(self.buf)
            self.buf.pop(0)
            self.buf.append(state)
            return buf_old

    def reset(self):
        """Reset the history sequence.

        Useful when you start a new episode.
        """
        self.buf = []

    def get_config(self):
        return {'history_length': self.history_length}


class AtariPreprocessor(Preprocessor):
    """Converts images to greyscale and downscales.

    Based on the preprocessing step described in:

    @article{mnih15_human_level_contr_throug_deep_reinf_learn,
    author =	 {Volodymyr Mnih and Koray Kavukcuoglu and David
                  Silver and Andrei A. Rusu and Joel Veness and Marc
                  G. Bellemare and Alex Graves and Martin Riedmiller
                  and Andreas K. Fidjeland and Georg Ostrovski and
                  Stig Petersen and Charles Beattie and Amir Sadik and
                  Ioannis Antonoglou and Helen King and Dharshan
                  Kumaran and Daan Wierstra and Shane Legg and Demis
                  Hassabis},
    title =	 {Human-Level Control Through Deep Reinforcement
                  Learning},
    journal =	 {Nature},
    volume =	 518,
    number =	 7540,
    pages =	 {529-533},
    year =	 2015,
    doi =        {10.1038/nature14236},
    url =	 {http://dx.doi.org/10.1038/nature14236},
    }

    You may also want to max over frames to remove flickering. Some
    games require this (based on animations and the limited sprite
    drawing capabilities of the original Atari).

    Parameters
    ----------
    new_size: 2 element tuple
      The size that each image in the state should be scaled to. e.g
      (84, 84) will make each image in the output have shape (84, 84).
    """
    def __init__(self, new_size):
        self.new_size = new_size

    def process_state_for_memory(self, state):
        """Scale, convert to greyscale and store as uint8.

        We don't want to save floating point numbers in the replay
        memory. We get the same resolution as uint8, but use a quarter
        to an eigth of the bytes (depending on float32 or float64)

        We recommend using the Python Image Library (PIL) to do the
        image conversions.
        """
        grey = self.convert_greyscale(state)
        grey_down = self.down_sample(grey)
        return grey_down.astype(np.uint8)

    def process_state_for_network(self, state):
        """Scale, convert to greyscale and store as float32.

        Basically same as process state for memory, but this time
        outputs float32 images.
        """
        grey = self.convert_greyscale(state)
        grey_down = self.down_sample(grey)
        return grey_down.astype(np.float32)

    def process_batch(self, samples):
        """The batches from replay memory will be uint8, convert to float32.

        Same as process_state_for_network but works on a batch of
        samples from the replay memory. Meaning you need to convert
        both state and next state values.
        """
        processed_samples = deepcopy(samples)
        for i in range(len(samples)):
            processed_samples[i].state = process_state_for_network(samples[i].state)
        return processed_samples

    def process_reward(self, reward):
        """Clip reward between -1 and 1."""
        return min(max(reward, -1), 1)

    def down_sample(self, state):
        return imresize(state, self.new_size)

    def convert_greyscale(self, color_mat):
        dtype = color_mat.dtype
        color_img = Image.fromarray(color_mat)
        grey_img = color_img.convert('L')
        grey_mat = np.fromstring(grey_img.tobytes(), dtype).reshape(color_mat.shape[:2])
        return grey_mat


class PreprocessorSequence(Preprocessor):
    """You may find it useful to stack multiple prepcrocesosrs (such as the History and the AtariPreprocessor).

    You can easily do this by just having a class that calls each preprocessor in succession.

    For example, if you call the process_state_for_network and you
    have a sequence of AtariPreproccessor followed by
    HistoryPreprocessor. This this class could implement a
    process_state_for_network that does something like the following:

    state = atari.process_state_for_network(state)
    return history.process_state_for_network(state)
    """
    def __init__(self, preprocessors):
        self.preprocessors = preprocessors

    def process_state_for_network(self, state):
        # MUST FOLLOW THIS
        for atari_proc in self.preprocessors[:-1]:
            state = atari_proc.process_state_for_network(state)
        hist_proc = self.preprocessors[-1]
        return hist_proc.process_state_for_network(state)
