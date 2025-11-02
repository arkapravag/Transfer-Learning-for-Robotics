import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import pybullet_envs_gymnasium  # registers Bullet envs
from evaluation_wrapper import UnifiedLocomotionEval

# =========================================================
# Observation Adapter: make MuJoCo obs match PyBullet-trained policy
# =========================================================
class ObservationAdapter(gym.ObservationWrapper):
    """
    Pads or truncates observation vectors so they match the PyBullet-trained
    PPO policy's expected dimension (22).
    """
    def __init__(self, env, target_dim):
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
            return np.concatenate([obs, np.zeros(self.target_dim - obs.shape[0], dtype=np.float32)])
        return obs


# =========================================================
# Load model
# =========================================================
MODEL_PATH = "../domain_randomization/ppo_walker_dr"   # path to trained PyBullet model
model = PPO.load(MODEL_PATH)
print(f"✅ Loaded model: {MODEL_PATH}")

# =========================================================
# Create Environments
# =========================================================
pyb_env = UnifiedLocomotionEval(gym.make("Walker2DBulletEnv-v0"))
mjc_env_raw = gym.make("Walker2d-v5")
mjc_env = UnifiedLocomotionEval(ObservationAdapter(mjc_env_raw, target_dim=22))


# =========================================================
# Evaluation Function
# =========================================================
def evaluate_env(env, n_episodes=10, name="env"):
    """
    Run model on the given environment and collect:
      - per-episode unified & original rewards
      - per-step unified reward trajectories
    """
    episode_unified_returns = []
    episode_env_returns = []
    trajectories = []  # list of per-step unified rewards per episode

    for ep in range(n_episodes):
        obs, info = env.reset()
        ep_unified_sum = 0.0
        ep_env_sum = 0.0
        unified_traj = []

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            unified_r = info.get("unified_reward", 0.0)
            ep_env_sum += reward
            ep_unified_sum += unified_r
            unified_traj.append(unified_r)
            done = terminated or truncated

        episode_unified_returns.append(ep_unified_sum)
        episode_env_returns.append(ep_env_sum)
        trajectories.append(unified_traj)
        print(f"{name} | Episode {ep+1}: unified={ep_unified_sum:.2f}, env={ep_env_sum:.2f}")

    # mean/std
    mean_unified = np.mean(episode_unified_returns)
    std_unified = np.std(episode_unified_returns)
    mean_env = np.mean(episode_env_returns)
    std_env = np.std(episode_env_returns)

    print(f"\n{name} Results over {n_episodes} episodes:")
    print(f"  Unified reward mean ± std : {mean_unified:.2f} ± {std_unified:.2f}")
    print(f"  Env reward mean ± std     : {mean_env:.2f} ± {std_env:.2f}\n")

    return mean_unified, mean_env, trajectories


# =========================================================
# Evaluate Both Simulators
# =========================================================
print("\n=== Evaluating on PyBullet (trained env) ===")
pyb_unified, pyb_envr, pyb_trajs = evaluate_env(pyb_env, n_episodes=10, name="PyBullet")

print("\n=== Evaluating on MuJoCo (zero-shot transfer) ===")
mjc_unified, mjc_envr, mjc_trajs = evaluate_env(mjc_env, n_episodes=10, name="MuJoCo")

# =========================================================
# Compare Zero-Shot Transfer
# =========================================================
drop = pyb_unified - mjc_unified
drop_pct = (drop / (abs(pyb_unified) + 1e-8)) * 100
print("=== Zero-Shot Transfer Summary ===")
print(f"PyBullet unified reward: {pyb_unified:.2f}")
print(f"MuJoCo unified reward  : {mjc_unified:.2f}")
print(f"Absolute drop          : {drop:.2f}")
print(f"Relative drop          : {drop_pct:.1f}%\n")


# =========================================================
# Plot Reward Trajectories
# =========================================================
def plot_trajectories(trajs_bullet, trajs_mujoco):
    """
    Plot mean unified reward trajectories per timestep for both simulators.
    """
    # Align lengths
    max_len = min(
        max(len(t) for t in trajs_bullet),
        max(len(t) for t in trajs_mujoco)
    )
    def pad_and_stack(trajs):
        padded = [t[:max_len] if len(t) >= max_len
                  else np.pad(t, (0, max_len - len(t))) for t in trajs]
        return np.array(padded)

    bullet_mat = pad_and_stack(trajs_bullet)
    mujoco_mat = pad_and_stack(trajs_mujoco)

    bullet_mean = bullet_mat.mean(axis=0)
    mujoco_mean = mujoco_mat.mean(axis=0)
    bullet_std = bullet_mat.std(axis=0)
    mujoco_std = mujoco_mat.std(axis=0)

    plt.figure(figsize=(8, 5))
    x = np.arange(max_len)
    plt.plot(x, bullet_mean, label="PyBullet (trained env)")
    plt.fill_between(x, bullet_mean - bullet_std, bullet_mean + bullet_std, alpha=0.2)
    plt.plot(x, mujoco_mean, label="MuJoCo (zero-shot)")
    plt.fill_between(x, mujoco_mean - mujoco_std, mujoco_mean + mujoco_std, alpha=0.2)
    plt.title("Unified Reward Trajectory Comparison")
    plt.xlabel("Timestep")
    plt.ylabel("Unified Reward (mean ± std)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("zero_shot_reward_comparison.png", dpi=200)
    plt.show()
    print("📈 Saved plot: zero_shot_reward_comparison.png")

plot_trajectories(pyb_trajs, mjc_trajs)
