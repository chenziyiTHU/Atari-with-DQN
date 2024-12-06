# Borrow a lot from labml.ai annotated deep learning paper implementations
# https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/rl/dqn/replay_buffer.py
import numpy as np
import random

class ReplayBuffer:
    '''Use Binary Segment Tree to store priority and experience replay'''
    def __init__(self, state_size, capacity, alpha):
        self.capacity = capacity
        self.alpha = alpha
        self.priority_sum = [0 for _ in range(2 * self.capacity)]
        self.priority_min = [float('inf') for _ in range(2 * self.capacity)]
        self.max_priority = 1.0
        self.data = {
            'state': np.zeros(shape=(capacity, *state_size), dtype=np.uint8),
            'action': np.zeros(shape=(capacity), dtype=np.int32),
            'reward': np.zeros(shape=(capacity), dtype=np.float32),
            'next_state': np.zeros(shape=(capacity, *state_size), dtype=np.uint8),
            'done': np.zeros(shape=(capacity), dtype=bool)
        }
        self.next_idx = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        idx = self.next_idx
        self.data['state'][idx] = state
        self.data['action'][idx] = action
        self.data['reward'][idx] = reward
        self.data['next_state'][idx] = next_state
        self.data['done'][idx] = done
        self.next_idx = (idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        priority_alpha = self.max_priority ** self.alpha
        self._set_priority_min(idx, priority_alpha)
        self._set_priority_sum(idx, priority_alpha)

    def _set_priority_min(self, idx, priority_alpha):
        '''Update the priority_min tree'''
        idx += self.capacity
        self.priority_min[idx] = priority_alpha
        while idx >= 2:
            idx //= 2
            self.priority_min[idx] = min(self.priority_min[2 * idx], self.priority_min[2 * idx + 1])

    def _set_priority_sum(self, idx, priority):
        '''Update the priority_sum tree'''
        idx += self.capacity
        self.priority_sum[idx] = priority
        while idx >= 2:
            idx //= 2
            self.priority_sum[idx] = self.priority_sum[2 * idx] + self.priority_sum[2 * idx + 1]

    def _sum(self):
        return self.priority_sum[1]

    def _min(self):
        return self.priority_min[1]

    def find_prefix_sum_idx(self, prefix_sum):
        '''Find largest idx such that sum(priority_sum[1:idx]) <= prefix_sum'''
        idx = 1
        while idx < self.capacity:
            if self.priority_sum[idx * 2] > prefix_sum:
                idx = idx * 2
            else:
                prefix_sum -= self.priority_sum[idx * 2]
                idx = idx * 2 + 1
        return idx - self.capacity

    def sample(self, batch_size, beta):
        samples = {
            'weights': np.zeros(shape=batch_size, dtype=np.float32),
            'indices': np.zeros(shape=batch_size, dtype=np.int32)
        }

        # Sample batch_size indices with probability propotional to $p_i^\alpha$
        for i in range(batch_size):
            p = random.random() * self._sum()
            idx = self.find_prefix_sum_idx(p)
            samples['indices'][i] = idx

        prob_min = self._min() / self._sum()
        max_weight = (prob_min * self.size) ** (-beta)

        for i in range(batch_size):
            idx = samples['indices'][i]
            prob = self.priority_sum[idx + self.capacity] / self._sum()
            weight = (prob * self.size) ** (-beta)
            samples['weights'][i] = weight / max_weight

        for k, v in self.data.items():
            samples[k] = v[samples['indices']]

        return samples

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.max_priority = max(self.max_priority, priority)
            priority_alpha = priority ** self.alpha
            self._set_priority_min(idx, priority_alpha)
            self._set_priority_sum(idx, priority_alpha)

    def is_full(self):
        return self.size == self.capacity

    def __len__(self):
        return self.size