from collections import deque


class Memory:
    def __init__(self, max_len):
        self.max_len = max_len
        self.frames = deque(maxlen=max_len)
        self.actions = deque(maxlen=max_len)
        self.rewards = deque(maxlen=max_len)
        self.done_flags = deque(maxlen=max_len)

    # xp means experience like in a Role Playing Game
    def add_xp(self, next_frame, next_frame_reward, next_action, next_done_flag):
        self.frames.append(next_frame)
        self.actions.append(next_action)
        self.rewards.append(next_frame_reward)
        self.done_flags.append(next_done_flag)
