"""replay_memory.py"""

import random
import collections
import numpy as np
import torch

class ReplayMemory(object):
    def __init__(self, max_size):   # 初始化经验回放样本池的容量大小
        self.buffer = collections.deque(maxlen=max_size)
        
    def append(self, exp):   # 经验池扩充函数
        self.buffer.append(exp)
    
    def sample(self, batch_size):   # 经验池采样函数，从堆栈中随机选取所需批次大小的样本并返回
        mini_batch = random.sample(self.buffer, batch_size)
        obs_batch, state_batch, action_batch, reward_batch, next_obs_batch, next_state_batch, done_batch = [], [], [], [], [], [], []
        for experience in mini_batch:
            o, s, a, r, n_o, n_s, done = experience
            obs_batch.append(o)
            state_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_obs_batch.append(n_o)
            next_state_batch.append(n_s)
            done_batch.append(done)

        return np.array(obs_batch).astype('float32'), np.array(state_batch).astype('float32'),\
            torch.tensor(action_batch).cpu().numpy().astype('float32'), torch.tensor(reward_batch).cpu().numpy().astype('float32'),\
            np.array(next_obs_batch).astype('float32'), np.array(next_state_batch).astype('float32'),\
            np.array(done_batch).astype('float32')

    def __len__(self):
        return len(self.buffer)