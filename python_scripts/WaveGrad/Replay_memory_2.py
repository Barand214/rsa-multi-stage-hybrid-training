"""Tai-stage replay memory helpers for WaveGrad training."""

import collections

import numpy as np
import torch


class ReplayMemory_2:
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)

    def append(self, exp):
        self.buffer.append(exp)

    def clear(self):
        self.buffer.clear()

    def sample(self, batch_size):
        mini_batch = list(self.buffer)
        obs_batch, state_batch, action_batch, reward_batch, done_batch = [], [], [], [], []
        for experience in mini_batch:
            obs, state, action, reward, done = experience
            obs_batch.append(obs)
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            done_batch.append(done)

        return (
            np.array(obs_batch).astype("float32"),
            np.array(state_batch).astype("float32"),
            torch.tensor(action_batch).cpu().numpy().astype("float32"),
            torch.tensor(reward_batch).cpu().numpy().astype("float32"),
            np.array(done_batch).astype("float32"),
        )

    def __len__(self):
        return len(self.buffer)
