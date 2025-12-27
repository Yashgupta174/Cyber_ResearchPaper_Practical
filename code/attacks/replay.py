# code/attacks/replay.py
class ReplayAttacker:
    def __init__(self, buffer_len=10, active=True):
        self.buffer = []
        self.buffer_len = buffer_len
        self.idx = 0
        self.active = active

    def feed(self, pkt):
        self.buffer.append(pkt.copy())
        if len(self.buffer) > self.buffer_len:
            self.buffer.pop(0)

    def attack_packet(self, pkt):
        if (not self.active) or len(self.buffer) == 0:
            return pkt
        out = self.buffer[self.idx % len(self.buffer)].copy()
        self.idx += 1
        return out
