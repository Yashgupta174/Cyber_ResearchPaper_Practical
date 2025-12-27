#!/usr/bin/env python3
"""
simulate_sql_like_attack.py

Simulates a structured ("SQL-like") FDI attack on a toy smart-grid,
trains a small tabular RL detector, and outputs:
  - sql_attack_plot.png   (eta vs time, log scale; marks attack start and detection)
  - sql_attack_traces.csv  (per timestep: t, eta, meas0, pred0, action)

Run:
  python3 simulate_sql_like_attack.py
Requires: numpy, matplotlib
Install: pip install numpy matplotlib
"""

import math, random, csv, os
import numpy as np
import matplotlib.pyplot as plt

# --- Simple Kalman implementation ---
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

# --- Simple Tabular Detector (SARSA-like) ---
class TabularDetector:
    def __init__(self, I=4, M=4, alpha=0.1, eps=0.1, c=0.2, beta=None):
        self.I = I; self.M = M; self.alpha = alpha; self.eps = eps; self.c = c
        self.Q = {}
        if beta is None:
            self.beta = [0.0] + [1e-3*(i+1) for i in range(I-1)] + [float('inf')]
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
            return random.choice([0,1])
        return int(self.Q[key].index(min(self.Q[key])))

    def update(self, window, a, r, nwindow, na):
        k1 = self.window_to_key(window); k2 = self.window_to_key(nwindow)
        if k1 not in self.Q: self.Q[k1] = [0.0,0.0]
        if k2 not in self.Q: self.Q[k2] = [0.0,0.0]
        target = r + self.Q[k2][na]
        self.Q[k1][a] += self.alpha * (target - self.Q[k1][a])

# ---------------- Simulation parameters ----------------
N = 5            # state dimension
K = 8            # number of meters
T = 200          # episode length
tau = 80         # attack starts at this time step

np.random.seed(1); random.seed(1)
H = np.random.randn(K, N) * 0.2

# --- baseline etas to compute quantization thresholds (beta) ---
kal_b = SimpleKalman(N)
x_true = np.zeros(N); x_est = np.zeros(N)
baseline_etas = []
for t in range(300):
    x_true = x_true + np.random.normal(0, 1e-3, N)
    y = H.dot(x_true) + np.random.normal(0, math.sqrt(kal_b.sigma_w), K)
    x_pred = kal_b.predict(x_est)
    x_est, _ = kal_b.update(x_pred, y, H)
    diff = y - H.dot(x_est)
    baseline_etas.append(float(np.dot(diff, diff)))
baseline_etas = np.array(baseline_etas)

# Beta thresholds: percentiles
I = 4
beta = [0.0] + [float(np.percentile(baseline_etas, 100.0*i/I)) for i in range(1, I)] + [float('inf')]

# initialize detector with baseline beta
det = TabularDetector(I=I, M=4, alpha=0.1, eps=0.15, c=0.2, beta=beta)

# Structured "SQL-like" FDI attack: b_t = H * g_t (small g_t but structured)
def generate_attack_b(t):
    g = np.ones(N) * 0.03
    g = g * (1.0 + 0.6 * math.sin(0.12 * t))
    return H.dot(g)

# run an episode; if train=True update detector
def run_episode(detector, train=False, seed=None):
    if seed is not None:
        np.random.seed(seed); random.seed(seed)
    kal = SimpleKalman(N)
    x_true = np.zeros(N); x_est = np.zeros(N)
    window = [0]*detector.M
    eta_trace=[]; meas0=[]; pred0=[]; actions=[]; detection_time=None
    for t in range(1, T+1):
        x_true = x_true + np.random.normal(0, 1e-3, N)
        y = H.dot(x_true) + np.random.normal(0, math.sqrt(kal.sigma_w), K)
        if t >= tau:
            y = y + generate_attack_b(t)
        x_pred = kal.predict(x_est)
        x_est, _ = kal.update(x_pred, y, H)
        diff = y - H.dot(x_est); eta = float(np.dot(diff,diff))
        eta_trace.append(eta); meas0.append(float(y[0])); pred0.append(float((H.dot(x_est))[0]))
        q = detector.quantize(eta)
        window.pop(0); window.append(q)
        a = detector.choose(window)
        actions.append(a)
        if a == 1:
            r = 1.0 if t < tau else 0.0
        else:
            r = detector.c if t >= tau else 0.0
        nwindow = list(window); na = detector.choose(nwindow)
        if train:
            detector.update(window, a, r, nwindow, na)
        if a == 1 and t >= tau and detection_time is None:
            detection_time = t
    return {"eta": np.array(eta_trace), "meas0": np.array(meas0), "pred0": np.array(pred0), "actions": np.array(actions), "detection_time": detection_time}

# Train detector briefly on attack episodes
for ep in range(300):
    _ = run_episode(det, train=True, seed=2000+ep)

# Evaluate one episode (no training)
res = run_episode(det, train=False, seed=9999)

# Prepare output files
out_dir = "results_sql_sim"
os.makedirs(out_dir, exist_ok=True)
png_path = os.path.join(out_dir, "sql_attack_plot.png")
csv_path = os.path.join(out_dir, "sql_attack_traces.csv")

# Plot
plt.figure(figsize=(11,5))
times = np.arange(1, T+1)
plt.plot(times, res["eta"], label="eta (residual norm)")
plt.axvline(tau, color='gray', linestyle='--', label="attack start (tau)")
if res["detection_time"] is not None:
    plt.axvline(res["detection_time"], color='red', linestyle='-', label=f"detection at t={res['detection_time']}")
plt.yscale('log'); plt.xlabel("Time step"); plt.ylabel("eta (log scale)")
plt.title("SQL-like structured FDI attack — residual eta and detection time")
plt.legend(); plt.grid(True, which="both", ls=":", linewidth=0.5)
plt.tight_layout(); plt.savefig(png_path, dpi=150); plt.close()

# Save traces CSV
with open(csv_path, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["t","eta","meas0","pred0","action"])
    for t in range(T):
        writer.writerow([t+1, float(res["eta"][t]), float(res["meas0"][t]), float(res["pred0"][t]), int(res["actions"][t])])

print("Simulation complete.")
print("Plot saved to:", png_path)
print("CSV traces saved to:", csv_path)
print("Detection time (None => no detection):", res["detection_time"])
