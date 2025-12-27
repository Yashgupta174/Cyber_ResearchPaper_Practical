
import os, json, math, random
import numpy as np
from pathlib import Path

# import from your code folder (ensure you run this from project root)
from detector import TabularDetector
from kalman import SimpleKalman
from sim_core import gen_measurement, single_episode
from attacks import dos as dos_mod
from attacks.replay import ReplayAttacker
from attacks.delay import DelayChannel

# --- Parameters (tweak as needed) ---
BASELINE_STEPS = 1000         # number of timesteps to collect baseline eta
N_BASE_EPISODES = 100         # alternative: run many short baseline episodes
TRAIN_EPISODES = 400          # training episodes per attack
EVAL_TRIALS = 50              # evaluation episodes per attack
TAU = 50                      # attack start time (same as other scripts)
T = 200                       # episode length
DETECTOR_PARAMS = dict(I=4, M=4, alpha=0.1, eps=0.1, c=0.2)
RESULTS_DIR = Path("results")
np.random.seed(0)
random.seed(0)

# --- Helpers (reuse the Kalman-like approach used in sim_core) ---
def collect_baseline_eta(N=5, K=8, steps=BASELINE_STEPS, noise_w=2e-4, sigma_v=1e-4):
    """Run a simple baseline simulation (no attack) and collect eta values."""
    H = np.random.randn(K, N) * 0.2
    kal = SimpleKalman(N, sigma_v=sigma_v, sigma_w=noise_w)
    x_true = np.zeros(N)
    x_est = np.zeros(N)
    etas = []
    for t in range(steps):
        x_true = x_true + np.random.normal(0, 1e-3, N)
        y = gen_measurement(x_true, H, math.sqrt(kal.sigma_w))
        x_pred = kal.predict(x_est)
        x_est, _ = kal.update(x_pred, y, H)
        diff = y - H.dot(x_est)
        eta = float(np.dot(diff, diff))
        etas.append(eta)
    return np.array(etas), H  # return H too so we can reuse consistent H if needed

def compute_beta_from_baseline(etas, I=4):
    """Compute simple quantization thresholds from baseline etas.
       We'll make thresholds around quantiles of baseline distribution.
    """
    # sort etas
    qs = [np.percentile(etas, p) for p in np.linspace(0, 100, I+1)]
    # ensure strictly increasing and finite; set last to inf
    beta = [float(max(0.0, qs[0]))]
    # We want beta[0] = 0, last = inf for our detector format
    beta = [0.0]
    for i in range(1, I):
        val = float(np.percentile(etas, 100.0 * i / I))
        beta.append(val)
    beta.append(float('inf'))
    return beta

def train_detector_on_attack(detector, attack_name, episodes=TRAIN_EPISODES, seed_base=1000):
    """Train the detector for `episodes` episodes where the given attack is active.
       This uses `single_episode` from sim_core to run episodes and update detector.
       For attacks that need objects (replay, delay), we create new instance per episode.
    """
    for e in range(episodes):
        seed = seed_base + e
        if attack_name == 'dos':
            _ = single_episode(detector, attack_type='dos', attack_params={'dos': dos_mod}, T=40, tau=5, seed=seed)
        elif attack_name == 'replay':
            attacker = ReplayAttacker(buffer_len=10)
            _ = single_episode(detector, attack_type='replay', attack_params={'attacker': attacker}, T=40, tau=5, seed=seed)
        elif attack_name == 'delay':
            chan = DelayChannel(base_delay=2.0, jitter=1.0)
            _ = single_episode(detector, attack_type='delay', attack_params={'chan': chan}, T=40, tau=5, seed=seed)
        else:
            raise ValueError("Unknown attack for training: " + attack_name)

def evaluate_detector(detector, attack_name, trials=EVAL_TRIALS, seed_base=2000):
    """Run `trials` evaluation episodes, return list of detection times (None if no detection)."""
    det_times = []
    for i in range(trials):
        seed = seed_base + i
        if attack_name == 'dos':
            t = single_episode(detector, attack_type='dos', attack_params={'dos': dos_mod}, T=T, tau=TAU, seed=seed)
        elif attack_name == 'replay':
            attacker = ReplayAttacker(buffer_len=10)
            t = single_episode(detector, attack_type='replay', attack_params={'attacker': attacker}, T=T, tau=TAU, seed=seed)
        elif attack_name == 'delay':
            chan = DelayChannel(base_delay=2.0, jitter=1.0)
            t = single_episode(detector, attack_type='delay', attack_params={'chan': chan}, T=T, tau=TAU, seed=seed)
        else:
            t = None
        det_times.append(t)
    return det_times

def compute_metrics_from_detection_times(det_times, tau=TAU, max_delay_accept=10):
    """Compute simple metrics (false alarms, avg delay for detections within bound).
       - false alarm: detection_time < tau
       - detection success: tau <= detection_time <= tau+max_delay_accept
    """
    n = len(det_times)
    false_alarms = sum(1 for t in det_times if t is not None and t < tau)
    detected = [t for t in det_times if t is not None and t >= tau and t <= tau + max_delay_accept]
    misses = sum(1 for t in det_times if t is None or (t is not None and t > tau + max_delay_accept))
    precision = len(detected) / (len(detected) + false_alarms + 1e-9)
    recall = len(detected) / (len(detected) + misses + 1e-9)
    fscore = 2 * precision * recall / (precision + recall + 1e-9)
    avg_delay = float(np.mean([t - tau for t in detected])) if len(detected) > 0 else None
    return {
        'n_trials': n,
        'false_alarms': int(false_alarms),
        'detections_within_bound': int(len(detected)),
        'misses': int(misses),
        'precision': float(precision),
        'recall': float(recall),
        'fscore': float(fscore),
        'avg_delay': avg_delay
    }

# -------------------------
# Main sequential workflow
# -------------------------
def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    # 1) Baseline collection
    print("Collecting baseline eta values (no attack)...")
    etas, H = collect_baseline_eta(steps=BASELINE_STEPS)
    baseline_dir = RESULTS_DIR / "baseline"
    baseline_dir.mkdir(exist_ok=True)
    np.save(baseline_dir / "eta_values.npy", etas)
    # 2) compute beta thresholds
    beta = compute_beta_from_baseline(etas, I=DETECTOR_PARAMS.get('I', 4))
    with open(baseline_dir / "beta.json", "w") as f:
        json.dump({'beta': beta, 'baseline_mean': float(np.mean(etas)), 'baseline_std': float(np.std(etas))}, f, indent=2)
    print("Baseline saved. Beta thresholds:", beta)

    # 3) Initialize detector with computed beta
    DET = TabularDetector(I=DETECTOR_PARAMS.get('I',4),
                         M=DETECTOR_PARAMS.get('M',4),
                         alpha=DETECTOR_PARAMS.get('alpha',0.1),
                         eps=DETECTOR_PARAMS.get('eps',0.1),
                         c=DETECTOR_PARAMS.get('c',0.2),
                         beta=beta)
    print("Initialized detector with baseline-informed beta.")

    # 4) For each attack: train and evaluate
    attack_list = ['dos', 'replay', 'delay']
    for attack in attack_list:
        print("\n=== Processing attack:", attack, "===")
        # Train detector on this attack
        print("Training detector on attack", attack, "for", TRAIN_EPISODES, "episodes...")
        train_detector_on_attack(DET, attack, episodes=TRAIN_EPISODES, seed_base=random.randint(1000,2000))
        # Evaluate
        print("Evaluating detector on attack", attack, "for", EVAL_TRIALS, "trials...")
        det_times = evaluate_detector(DET, attack, trials=EVAL_TRIALS, seed_base=random.randint(2000,3000))
        metrics = compute_metrics_from_detection_times(det_times, tau=TAU)
        # Save outputs
        attack_dir = RESULTS_DIR / attack
        attack_dir.mkdir(exist_ok=True)
        with open(attack_dir / "eval_summary.json", "w") as f:
            json.dump({'detection_times': det_times, 'metrics': metrics}, f, indent=2)
        print("Saved evaluation summary to", str(attack_dir / "eval_summary.json"))
        print("Metrics:", metrics)
        # Optional: you can reset or continue training for next attack.
        # Here we continue training cumulatively (the detector keeps learning).
        # If you prefer fresh detector per attack, re-initialize DET before next loop.

    print("\nSequential runs complete. Check results/ folder for baseline and per-attack summaries.")

if __name__ == "__main__":
    main()
