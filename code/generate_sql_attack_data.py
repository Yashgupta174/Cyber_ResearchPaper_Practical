import pandas as pd
import numpy as np

# --- Simulation parameters ---
np.random.seed(42)
time = np.arange(0, 200, 1)

# Normal (pre-attack): t < 80
residual_normal = np.random.normal(0.01, 0.002, 80)

# Attack phase (80–130): sudden spike & instability
residual_attack = np.random.normal(0.05, 0.015, 50)

# Post-attack recovery (130–199): gradually stabilizing
residual_recovery = np.linspace(0.05, 0.015, 70) + np.random.normal(0, 0.003, 70)

# Combine all residuals
residual = np.concatenate([residual_normal, residual_attack, residual_recovery])

# Define flags
attack_flag = [0 if t < 80 else 1 if t < 130 else 0 for t in time]
detection_flag = [1 if t == 85 else 0 for t in time]  # detection happens shortly after attack starts

# System state labels
system_state = []
for t in time:
    if t < 80:
        system_state.append("normal")
    elif 80 <= t < 130:
        system_state.append("attack")
    else:
        system_state.append("recovered")

# Measured vs Predicted voltage (Kalman-like effect)
measured_voltage = 1.0 + np.random.normal(0, 0.01, len(time))
predicted_voltage = measured_voltage - (residual * np.random.uniform(0.5, 1.2))

# Create DataFrame
df = pd.DataFrame({
    "time": time,
    "residual": residual,
    "attack_flag": attack_flag,
    "detection_flag": detection_flag,
    "system_state": system_state,
    "measured_voltage": measured_voltage,
    "predicted_voltage": predicted_voltage
})

# Save to CSV
df.to_csv("sql_attack_trace_data.csv", index=False)
print("✅ Sample file 'sql_attack_trace_data.csv' generated successfully!")
