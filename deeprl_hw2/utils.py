import cv2
import numpy as np


def preprocess_frame(input_frame, output_size=84):
    gray = cv2.cvtColor(input_frame, cv2.COLOR_RGB2GRAY)
    output = cv2.resize(gray, (output_size, output_size))
    output = output.astype(np.float32)
    return output
