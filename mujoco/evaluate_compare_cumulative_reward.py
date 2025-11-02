import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import pybullet_envs_gymnasium  # registers Bullet envs
from evaluation_wrapper import UnifiedLocomotionEval


# =========================================================
# Observation Adapter (17-dim MuJoCo → 22-dim PyBullet)
# =========================================================
class ObservationAdapter(gym.ObservationWrapper):
    def __init__(self, env, target_dim=22):
        super().__init__(env)
        self.target_dim = target_dim
        low = np.concatenate([
            env.observation_space.low,
            np.full(max(0, target_dim - env.observation_space.shape[0]), -np.inf)
        ])
        high = np.concatenate([
            env.observation_space.high,
            np.full(max(0, target_dim - env.observation_space.shape[0]), np.inf)
        ])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs):
        if obs.shape[0] > self.target_dim:
            return obs[:self.target_dim]
        elif obs.shape[0] < self.target_dim:
            return np.concatenate(
                [obs, np.zeros(self.target_dim - obs.shape[0], dtype=np.float32)]
            )
        return obs


# =========================================================
# Load Domain-Randomized PPO model
# =========================================================
MODEL_PATH = "../domain_randomization/ppo_walker_dr_full"
# MODEL_PATH = "../ppo_pybullet_walker"
model = PPO.load(MODEL_PATH)
print(f"✅ Loaded model: {MODEL_PATH}")

# =========================================================
# Create Environments
# =========================================================
pyb_env = UnifiedLocomotionEval(gym.make("Walker2DBulletEnv-v0"))
mjc_env = UnifiedLocomotionEval(ObservationAdapter(gym.make("Walker2d-v5"), target_dim=22))


# =========================================================
# Evaluation for fixed number of timesteps
# =========================================================
def evaluate_cumulative(env, total_steps=10_000, name="env"):
    """
    Runs the policy for a fixed number of timesteps (total_steps),
    summing unified and native rewards across all episodes within that window.
    """
    obs, info = env.reset()
    cumulative_unified = 0.0
    cumulative_env = 0.0
    unified_traj = []

    steps = 0
    while steps < total_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        unified_r = info.get("unified_reward", 0.0)

        cumulative_env += reward
        cumulative_unified += unified_r
        unified_traj.append(unified_r)
        steps += 1

        if terminated or truncated:
            obs, info = env.reset()

    print(f"\n{name} cumulative evaluation ({total_steps} steps):")
    print(f"  Total unified reward: {cumulative_unified:.2f}")
    print(f"  Total native reward : {cumulative_env:.2f}")
    print(f"  Mean unified/step   : {cumulative_unified/total_steps:.3f}")
    print(f"  Mean native/step    : {cumulative_env/total_steps:.3f}\n")

    return cumulative_unified, cumulative_env, unified_traj


# =========================================================
# Run cumulative evaluations
# =========================================================
print("\n=== Evaluating Domain-Randomized PPO on PyBullet ===")
pyb_cum_unified, pyb_cum_env, pyb_traj = evaluate_cumulative(pyb_env, total_steps=10_000, name="PyBullet")

print("\n=== Evaluating Domain-Randomized PPO on MuJoCo (Zero-Shot) ===")
mjc_cum_unified, mjc_cum_env, mjc_traj = evaluate_cumulative(mjc_env, total_steps=10_000, name="MuJoCo")

# =========================================================
# Compare Zero-Shot Cumulative Results
# =========================================================
abs_drop = pyb_cum_unified - mjc_cum_unified
rel_drop = (abs_drop / (abs(pyb_cum_unified) + 1e-8)) * 100

print("=== Zero-Shot Cumulative Transfer Summary ===")
print(f"PyBullet total unified reward: {pyb_cum_unified:.2f}")
print(f"MuJoCo total unified reward  : {mjc_cum_unified:.2f}")
print(f"Absolute drop                : {abs_drop:.2f}")
print(f"Relative drop                : {rel_drop:.1f}%\n")


# =========================================================
# Plot per-step unified rewards
# =========================================================
def plot_cumulative(pyb_traj, mjc_traj):
    # Trim to same length
    n = min(len(pyb_traj), len(mjc_traj))
    pyb_traj, mjc_traj = np.array(pyb_traj[:n]), np.array(mjc_traj[:n])
    # Compute cumulative sums
    pyb_cum = np.cumsum(pyb_traj)
    mjc_cum = np.cumsum(mjc_traj)

    plt.figure(figsize=(8, 5))
    plt.plot(pyb_cum, label="PyBullet (trained env)")
    plt.plot(mjc_cum, label="MuJoCo (zero-shot)")
    plt.title("Cumulative Unified Reward (10 000 steps)")
    plt.xlabel("Timestep")
    plt.ylabel("Cumulative Unified Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("cumulative_zero_shot_comparison.png", dpi=200)
    plt.show()
    print("📈 Saved plot: cumulative_zero_shot_comparison.png")

plot_cumulative(pyb_traj, mjc_traj)
