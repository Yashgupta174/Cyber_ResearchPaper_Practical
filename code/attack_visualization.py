import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load your attack data (e.g., from "sql_attack_trace_data.csv")
# Columns expected: time, residual, attack_flag, detection_flag, system_state
df = pd.read_csv("sql_attack_trace_data.csv")

# Extract columns
time = df["time"]
residual = df["residual"]
attack_flag = df["attack_flag"]
detection_flag = df["detection_flag"]
system_state = df["system_state"]

# Identify key points
attack_start = time[attack_flag == 1].min()
detection_time = time[detection_flag == 1].min()
recovery_time = time[system_state == "recovered"].min() if "recovered" in system_state.values else None

# 1️⃣ Residual vs Time (shows pre, during, post attack)
plt.figure(figsize=(10,6))
plt.plot(time, residual, label="Residual (ηt)", color='blue')
plt.axvline(attack_start, color='red', linestyle='--', label='Attack Start')
plt.axvline(detection_time, color='orange', linestyle='--', label='Detection Time')
if recovery_time:
    plt.axvline(recovery_time, color='green', linestyle='--', label='Recovery')

plt.title("System Residuals Before, During, and After Attack")
plt.xlabel("Time (t)")
plt.ylabel("Residual ηt (deviation)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 2️⃣ Detector Action Timeline
plt.figure(figsize=(10,3))
plt.plot(time, detection_flag, color='purple', drawstyle='steps-post')
plt.title("RL Detector Action Timeline (0=Continue, 1=Attack Detected)")
plt.xlabel("Time")
plt.ylabel("Action")
plt.grid(True)
plt.tight_layout()
plt.show()

# 3️⃣ System State Visualization
states_color = {"normal": "green", "attack": "red", "recovered": "blue"}
colors = df["system_state"].map(states_color)

plt.figure(figsize=(10,3))
plt.scatter(time, [1]*len(time), c=colors)
plt.title("System State (Green=Normal, Red=Attack, Blue=Recovered)")
plt.yticks([])
plt.xlabel("Time")
plt.tight_layout()
plt.show()
