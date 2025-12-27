# code/run_replay.py
import os, json
from detector import TabularDetector
from attacks.replay import ReplayAttacker
from sim_core import single_episode

def run_replay(n_runs=8, buffer_len=10, seed_base=2000):
    os.makedirs('results/replay', exist_ok=True)
    metrics_list = []
    for i in range(1, n_runs+1):
        print(f"[Replay] Run {i}/{n_runs}")
        det = TabularDetector(I=4, M=4, alpha=0.1, eps=0.1, c=0.2)
        attacker = ReplayAttacker(buffer_len=buffer_len)
        # warm-up training (feed some episodes)
        for _ in range(400):
            _ = single_episode(det, attack_type='replay', attack_params={'attacker': attacker}, T=50, tau=5, seed=seed_base + i + _)
        # Evaluate
        attacker = ReplayAttacker(buffer_len=buffer_len)  # fresh attacker for evaluation
        detection_time = single_episode(det, attack_type='replay', attack_params={'attacker': attacker}, T=200, tau=50, seed=seed_base + i)
        run_dir = f'results/replay/run_{i}'
        os.makedirs(run_dir, exist_ok=True)
        metrics = {
            'run': i,
            'buffer_len': buffer_len,
            'detection_time': detection_time
        }
        with open(f'{run_dir}/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        metrics_list.append(metrics)
    with open('results/replay/summary.json', 'w') as f:
        json.dump(metrics_list, f, indent=2)
    print("Replay runs complete. Summary saved to results/replay/summary.json")

if __name__ == '__main__':
    run_replay(n_runs=8, buffer_len=10)
