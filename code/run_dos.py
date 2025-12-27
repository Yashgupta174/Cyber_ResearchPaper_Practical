# code/run_dos.py
import os, json
from detector import TabularDetector
from attacks import dos
from sim_core import single_episode
import numpy as np, random

def run_dos(n_runs=8, p_drop=0.2, seed_base=1000):
    os.makedirs('results/dos', exist_ok=True)
    metrics_list = []
    for i in range(1, n_runs+1):
        print(f"[DoS] Run {i}/{n_runs}")
        det = TabularDetector(I=4, M=4, alpha=0.1, eps=0.1, c=0.2)
        # warm-up training (small number of episodes) to populate Q-table
        for _ in range(400):
            _ = single_episode(det, attack_type='dos', attack_params={'dos': dos}, T=50, tau=5, seed=seed_base + i + _)
        # now evaluate on a standard episode
        detection_time = single_episode(det, attack_type='dos', attack_params={'dos': dos}, T=200, tau=50, seed=seed_base + i)
        # Save per-run metrics
        run_dir = f'results/dos/run_{i}'
        os.makedirs(run_dir, exist_ok=True)
        metrics = {
            'run': i,
            'p_drop': p_drop,
            'detection_time': detection_time
        }
        with open(f'{run_dir}/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        metrics_list.append(metrics)
    # aggregate summary
    with open('results/dos/summary.json', 'w') as f:
        json.dump(metrics_list, f, indent=2)
    print("DoS runs complete. Summary saved to results/dos/summary.json")

if __name__ == '__main__':
    run_dos(n_runs=8, p_drop=0.2)
