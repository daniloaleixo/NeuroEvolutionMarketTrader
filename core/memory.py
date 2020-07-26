import heapq
import numpy as np
from itertools import count
from collections import deque
tiebreaker = count()



class ReplayMemory():
    """ Standard replay memory sampled uniformly """
    def __init__(self, max_size=10000):
        self.memory = deque()
        self.max_size = max_size

    def add(self, transition, loss=0):
        self.memory.append(transition)
        if len(self.memory) > self.max_size:
            self.memory.popleft()

    def batch(self, n):
        return [ self._random_episode() for i in range(n) ]

    def _random_episode(self):
        rand_ind = np.random.randint(0, self.size())
        return self.memory[rand_ind]

    def size(self):
        return len(self.memory)

    def is_full(self):
        return True if self.size() >= self.max_size else False


class PER():
    """ Prioritized replay memory using binary heap """
    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.memory = []

    def add(self, transition, TDerror):
        heapq.heappush(self.memory, (-TDerror, next(tiebreaker), transition))
        if self.size() > self.max_size:
            self.memory = self.memory[:-1]
        heapq.heapify(self.memory)
    
    def batch(self, n):
        batch = heapq.nsmallest(n, self.memory)
        batch = [e for (_, _, e) in batch]
        self.memory = self.memory[n:]
        return batch

    def size(self):
        return len(self.memory)

    def is_full(self):
        return True if self.size() >= self.max_size else False