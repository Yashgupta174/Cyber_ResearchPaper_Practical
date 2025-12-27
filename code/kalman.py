# code/kalman.py
import numpy as np

class SimpleKalman:
    def __init__(self, N, sigma_v=1e-4, sigma_w=2e-4):
        self.N = N
        self.sigma_v = sigma_v
        self.sigma_w = sigma_w
        self.A = np.eye(N)
        self.F = np.eye(N) * 1e-3

    def predict(self, x_prev):
        x_pred = self.A.dot(x_prev)
        self.F = self.A.dot(self.F).dot(self.A.T) + self.sigma_v * np.eye(self.N)
        return x_pred

    def update(self, x_pred, y, H):
        S = H.dot(self.F).dot(H.T) + self.sigma_w * np.eye(y.shape[0])
        K = self.F.dot(H.T).dot(np.linalg.inv(S))
        x_upd = x_pred + K.dot(y - H.dot(x_pred))
        self.F = (np.eye(self.N) - K.dot(H)).dot(self.F)
        return x_upd, K
