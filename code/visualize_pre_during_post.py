#!/usr/bin/env python3
"""
visualize_pre_during_post.py

Inputs:
  - sql_attack_trace_data.csv (expected columns: time,residual,attack_flag,detection_flag,system_state,
      measured_voltage,predicted_voltage)
  - Optionally per-meter residual columns like resid_meter_0, resid_meter_1, ... for heatmap.

Outputs (saved to ./plots/):
  - residual_vs_time.png
  - meas_vs_pred_meter0.png
  - action_timeline.png
  - (optional) per_meter_heatmap.png
"""
import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CSV_NAME = "sql_attack_trace_data.csv"
OUT_DIR = Path("plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load CSV
if not Path(CSV_NAME).exists():
    print(f"Error: {CSV_NAME} not found in current folder ({Path.cwd()}).")
    sys.exit(1)

df = pd.read_csv(CSV_NAME)

# Basic checks / fallback names
if 'time' not in df.columns:
    # try common alternatives
    if 't' in df.columns: df = df.rename(columns={'t':'time'})
    else:
        df.insert(0, 'time', np.arange(len(df)))

# required columns fallback
if 'residual' not in df.columns:
    # try computing residual if measured/predicted available
    if 'measured_voltage' in df.columns and 'predicted_voltage' in df.columns:
        df['residual'] = (df['measured_voltage'] - df['predicted_voltage'])**2
    else:
        raise ValueError("CSV must contain 'residual' or both measured/predicted columns.")

# derive attack/detection/recovery points
time = df['time'].to_numpy()
residual = df['residual'].to_numpy()

attack_times = df.loc[df.get('attack_flag', 0)==1, 'time'].to_numpy() if 'attack_flag' in df.columns else np.array([])
detection_times = df.loc[df.get('detection_flag', 0)==1, 'time'].to_numpy() if 'detection_flag' in df.columns else np.array([])
# recovery: find first 'recovered' state if available
recovery_times = np.array([])
if 'system_state' in df.columns:
    recovered = df[df['system_state'].astype(str).str.lower() == 'recovered']
    if not recovered.empty:
        recovery_times = recovered['time'].to_numpy()

# Helper to pick first occurrences
attack_start = int(attack_times.min()) if attack_times.size>0 else None
detection_time = int(detection_times.min()) if detection_times.size>0 else None
recovery_time = int(recovery_times.min()) if recovery_times.size>0 else None

# ---------- Plot 1: Residual vs Time ----------
plt.figure(figsize=(11,5))
plt.plot(time, residual, label='Residual (η)', linewidth=1.2)
if attack_start is not None:
    plt.axvline(attack_start, color='red', linestyle='--', label=f'Attack start (t={attack_start})')
if detection_time is not None:
    plt.axvline(detection_time, color='orange', linestyle='--', label=f'Detection (t={detection_time})')
if recovery_time is not None:
    plt.axvline(recovery_time, color='green', linestyle='--', label=f'Recovery (t={recovery_time})')
plt.yscale('log')  # log scale often helps
plt.xlabel('Time step')
plt.ylabel('Residual η (log scale)')
plt.title('Residual (η) — Pre / During / Post attack')
plt.grid(True, which='both', ls=':', lw=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "residual_vs_time.png", dpi=150)
plt.close()

# ---------- Plot 2: Measured vs Predicted (meter 0) ----------
# we expect measured_voltage and predicted_voltage or meas0/pred0
meas_col = None
pred_col = None
for cand in ['measured_voltage','meas0','measurement','measured']:
    if cand in df.columns:
        meas_col = cand; break
for cand in ['predicted_voltage','pred0','pred','prediction']:
    if cand in df.columns:
        pred_col = cand; break

if meas_col and pred_col:
    plt.figure(figsize=(11,5))
    plt.plot(time, df[meas_col], label='Measured (meter 0)', linewidth=1.0)
    plt.plot(time, df[pred_col], label='Kalman predicted (meter 0)', linewidth=1.0, linestyle='--')
    if attack_start is not None:
        plt.axvline(attack_start, color='red', linestyle='--')
    if detection_time is not None:
        plt.axvline(detection_time, color='orange', linestyle='--')
    plt.xlabel('Time step')
    plt.ylabel('Value')
    plt.title('Measured vs Kalman Predicted (meter 0)')
    plt.legend()
    plt.grid(True, ls=':', lw=0.5)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "meas_vs_pred_meter0.png", dpi=150)
    plt.close()
else:
    print("Note: measured/predicted columns not found; skipping meas-vs-pred plot.")

# ---------- Plot 3: Detector action timeline ----------
# Look for an action column; fallback to detection_flag (0/1)
action_col = None
for cand in ['action','detection_flag','detected','alarm']:
    if cand in df.columns:
        action_col = cand; break

if action_col:
    plt.figure(figsize=(11,2.5))
    y = df[action_col].astype(int)
    plt.step(time, y, where='post', linewidth=1.5)
    plt.ylim(-0.1, 1.1)
    plt.xlabel('Time step')
    plt.yticks([0,1], ['Continue (0)','Detected (1)'])
    plt.title('Detector action timeline')
    plt.grid(True, ls=':', lw=0.4)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "action_timeline.png", dpi=150)
    plt.close()
else:
    print("Note: no action column found; skipping action timeline plot.")

# ---------- Optional Plot 4: per-meter residual heatmap ----------
# Look for columns named resid_meter_0, resid_meter_1, ...
resid_cols = [c for c in df.columns if str(c).lower().startswith('resid_meter') or str(c).lower().startswith('resid_m')]
if resid_cols:
    mat = df[resid_cols].to_numpy().T  # shape (n_meters, n_time)
    fig, ax = plt.subplots(figsize=(11,4))
    im = ax.imshow(mat, aspect='auto', cmap='viridis', origin='lower')
    ax.set_ylabel('Meter index')
    ax.set_xlabel('Time step')
    ax.set_title('Per-meter residuals heatmap (rows=meters, cols=time)')
    plt.colorbar(im, ax=ax, label='residual')
    if attack_start is not None:
        ax.axvline(attack_start, color='red', linestyle='--')
    plt.tight_layout()
    plt.savefig(OUT_DIR / "per_meter_heatmap.png", dpi=150)
    plt.close()
else:
    # maybe user has per-meter measurements; try compute per-meter residuals if H/pred available -- skip now
    pass

print(f"Plots saved in folder: {OUT_DIR.resolve()}")
if attack_start is None:
    print("Warning: no attack_flag found in CSV; pre/during/post boundaries may not be marked.")
if detection_time is None:
    print("Warning: no detection_flag found in CSV; detection line won't be shown.")
