# code/attacks/delay.py
import random
from collections import deque

class DelayChannel:
    def __init__(self, base_delay=2.0, jitter=1.0):
        self.base = base_delay
        self.jitter = jitter
        self.queue = deque()

    def send(self, pkt, t):
        delay = self.base + (random.random()*2 - 1) * self.jitter
        release_time = t + max(0.0, delay)
        self.queue.append((release_time, pkt))

    def receive(self, t):
        out = []
        while self.queue and self.queue[0][0] <= t:
            out.append(self.queue.popleft()[1])
        return out
