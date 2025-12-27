# code/sim_core.py
import math, random
import numpy as np
from kalman import SimpleKalman

def gen_measurement(true_state, H, noise_std):
    return H.dot(true_state) + np.random.normal(0, noise_std, H.shape[0])

def single_episode(detector, attack_type=None, attack_params=None, T=200, tau=100, seed=None):


    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    N = 5   # small model for speed
    K = 8
    H = np.random.randn(K, N) * 0.2
    kal = SimpleKalman(N)
    x_true = np.zeros(N)
    x_est = np.zeros(N)
    window = [0] * detector.M

    for t in range(1, T+1):
        x_true = x_true + np.random.normal(0, 1e-3, N)
        y = gen_measurement(x_true, H, math.sqrt(kal.sigma_w))

        # Attack injection logic
        if attack_type == 'dos' and t >= tau:
            recv = attack_params['dos'].dos_apply(y)  # dos_apply returns None or packet
        elif attack_type == 'replay':
            # always feed buffer to maintain content
            if 'replay' not in attack_params:
                attack_params['replay'] = None
            if t < tau:
                attack_params['attacker'].feed(y)
                recv = y
            else:
                attack_params['attacker'].feed(y)
                recv = attack_params['attacker'].attack_packet(y)
        elif attack_type == 'delay':
            if t >= tau:
                attack_params['chan'].send(y, t)
                recs = attack_params['chan'].receive(t)
                recv = recs[-1] if recs else None
            else:
                recv = y
        else:
            recv = y

        # Kalman predict/update
        x_pred = kal.predict(x_est)
        if recv is not None:
            x_est, _ = kal.update(x_pred, recv, H)
        else:
            x_est = x_pred

        # compute eta
        eta = 0.0
        if recv is not None:
            diff = recv - H.dot(x_est)
            eta = float(np.dot(diff, diff))

        # quantize & update window
        q = detector.quantize(eta)
        window.pop(0); window.append(q)

        # choose action & compute cost (for SARSA updates we need next state/action)
        a = detector.choose(window)   # 0 continue, 1 stop
        # compute cost
        if a == 1:
            r = 1.0 if t < tau else 0.0
        else:
            r = detector.c if t >= tau else 0.0

        # simulate next action for sarsa-like update
        nwindow = list(window)
        na = detector.choose(nwindow)
        detector.update(window, a, r, nwindow, na)

        if a == 1 and t >= tau:
            return t  # detection time

    return None
