import cv2
import numpy as np


def resize_frame(frame, state_dim_1, state_dim_2):
    frame = frame[8:-1, 4:-48]
    frame = np.average(frame, axis=2)
    frame = cv2.resize(frame, (state_dim_2, state_dim_1), interpolation=cv2.INTER_NEAREST)
    frame = np.array(frame, dtype=np.uint8)

    return frame
