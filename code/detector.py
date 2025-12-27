# code/detector.py
import random

class TabularDetector:
    """
    Tabular SARSA-like detector:
    - quantize scalar eta via beta thresholds
    - sliding window of size M -> tuple key
    - Q stored in dict: key -> [Q_continue, Q_stop]
    """
    def __init__(self, I=4, M=4, alpha=0.1, eps=0.1, c=0.2, beta=None):
        self.I = I
        self.M = M
        self.alpha = alpha
        self.eps = eps
        self.c = c
        self.Q = {}
        # default beta thresholds (tuneable). Must have length I+1 with beta[0]=0, beta[-1]=inf
        if beta is None:
            self.beta = [0.0, 9.5e-4, 1.05e-3, 1.15e-3, float('inf')]
        else:
            self.beta = beta

    def quantize(self, eta):
        for i in range(1, len(self.beta)):
            if eta < self.beta[i]:
                return i-1
        return len(self.beta)-2

    def window_to_key(self, window):
        return tuple(window)

    def choose(self, window):
        key = self.window_to_key(window)
        if key not in self.Q:
            self.Q[key] = [0.0, 0.0]
        if random.random() < self.eps:
            return random.choice([0, 1])  # 0=continue, 1=stop
        return int(self.Q[key].index(min(self.Q[key])))

    def update(self, window, a, r, nwindow, na):
        k1 = self.window_to_key(window)
        k2 = self.window_to_key(nwindow)
        if k1 not in self.Q: self.Q[k1] = [0.0, 0.0]
        if k2 not in self.Q: self.Q[k2] = [0.0, 0.0]
        target = r + self.Q[k2][na]
        self.Q[k1][a] += self.alpha * (target - self.Q[k1][a])
