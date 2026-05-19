"""Replay memory helpers for WaveGrad reward-weighted diffusion training."""

import collections
import random

import numpy as np
import torch


class ReplayMemory(object):
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)

    def append(self, exp):
        self.buffer.append(exp)

    def clear(self):
        self.buffer.clear()

    def sample(self, batch_size):
        if len(self.buffer) <= batch_size:
            mini_batch = list(self.buffer)
        else:
            mini_batch = random.sample(self.buffer, batch_size)

        obs_batch, state_batch, action_batch, reward_batch, done_batch, value_batch = [], [], [], [], [], []
        for experience in mini_batch:
            obs, state, action, reward, done, value = experience
            obs_batch.append(obs)
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            done_batch.append(done)
            value_batch.append(value)

        return (
            np.array(obs_batch),
            np.array(state_batch).astype("float32"),
            torch.tensor(action_batch, dtype=torch.float32),
            torch.tensor(reward_batch, dtype=torch.float32),
            torch.tensor(done_batch, dtype=torch.float32),
            torch.tensor(value_batch, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)
