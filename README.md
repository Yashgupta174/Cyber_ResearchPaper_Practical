A. System State Estimation Using Kalman Filter
B. Observation Processing (Quantization + Sliding Window)
C. Model-Free Reinforcement Learning (SARSA Algorithm)
D. Attack Simulation
E. Online Detection

mrthododlogy text for visio diagram

START
   ↓
Smart Grid Simulation (Normal + Attack data)
   ↓
Meter Measurements (y_t)
   ↓
Kalman Filter
   ↓
Predicted Measurement (y_pred_t)
   ↓
Residual Computation (η_t = |y_t - y_pred_t|)
   ↓
Quantization of η_t → θ_i
   ↓
Sliding Window (Observation O_t)
   ↓
[TRAINING PHASE]
   ↓
SARSA RL Agent learns Q(O_t, action)
   ↓
[ONLINE PHASE]
   ↓
Use Q-table for decision
   ↓
IF action = STOP → Attack Detected
ELSE → Continue
   ↓
END


Algorithms  

Algorithm 1: Residual Generation (Kalman Filter)
Input: Measurements y_t
Output: Residual η_t

1. Predict state x̂_t using Kalman filter
2. Predict measurement y_pred_t = H × x̂_t
3. Compute residual η_t = |y_t - y_pred_t|
4. Return η_t

Algorithm 2: RL Training (SARSA)
1. Initialize Q-table
2. For each episode:
     a. Reset environment (normal or attack)
     b. For each time step:
         i.  Collect η_t
         ii. Quantize η_t → θ_i
         iii.Update sliding window O_t
         iv. Choose action using ε-greedy
         v. Receive cost from environment
         vi.Update Q(O_t, action)
3. Save Q-table

Algorithm 3: Online Detection
1. Load trained Q-table
2. For each time step:
     a. Compute η_t
     b. Quantize and update sliding window
     c. Select action = argmin Q(O_t, action)
     d. If action = STOP → declare attack
