# code/run_delay.py
import os, json
from detector import TabularDetector
from attacks.delay import DelayChannel
from sim_core import single_episode

def run_delay(n_runs=8, base_delay=2.0, jitter=1.0, seed_base=3000):
    os.makedirs('results/delay', exist_ok=True)
    metrics_list = []
    for i in range(1, n_runs+1):
        print(f"[Delay] Run {i}/{n_runs}")
        det = TabularDetector(I=4, M=4, alpha=0.1, eps=0.1, c=0.2)
        chan = DelayChannel(base_delay=base_delay, jitter=jitter)
        # warm-up training
        for _ in range(400):
            _ = single_episode(det, attack_type='delay', attack_params={'chan': chan}, T=50, tau=5, seed=seed_base + i + _)
        # fresh channel for evaluation
        chan = DelayChannel(base_delay=base_delay, jitter=jitter)
        detection_time = single_episode(det, attack_type='delay', attack_params={'chan': chan}, T=200, tau=50, seed=seed_base + i)
        run_dir = f'results/delay/run_{i}'
        os.makedirs(run_dir, exist_ok=True)
        metrics = {
            'run': i,
            'base_delay': base_delay,
            'jitter': jitter,
            'detection_time': detection_time
        }
        with open(f'{run_dir}/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        metrics_list.append(metrics)
    with open('results/delay/summary.json', 'w') as f:
        json.dump(metrics_list, f, indent=2)
    print("Delay runs complete. Summary saved to results/delay/summary.json")

if __name__ == '__main__':
    run_delay(n_runs=8, base_delay=2.0, jitter=1.0)
