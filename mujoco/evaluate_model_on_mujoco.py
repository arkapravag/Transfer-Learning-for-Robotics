import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from evaluation_wrapper import UnifiedLocomotionEval   # same file you used before

# =========================================================
# Observation Adapter to match MuJoCo obs (17) → PyBullet-trained (22)
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
# Load trained domain-randomized PPO model
# =========================================================
MODEL_PATH = "../domain_randomization/ppo_walker_dr.zip"  # make sure this file exists
model = PPO.load(MODEL_PATH)
print(f"✅ Loaded domain-randomized model: {MODEL_PATH}")

# =========================================================
# Create MuJoCo environment (wrapped)
# =========================================================
mjc_env_raw = gym.make("Walker2d-v5")
mjc_env = UnifiedLocomotionEval(ObservationAdapter(mjc_env_raw, target_dim=22))


# =========================================================
# Evaluation Function
# =========================================================
def evaluate_env(env, n_episodes=10, name="env"):
    """
    Run model on given environment and collect per-episode unified rewards
    and reward trajectories for plotting.
    """
    episode_unified_returns = []
    episode_env_returns = []
    trajectories = []

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

    mean_unified = np.mean(episode_unified_returns)
    std_unified = np.std(episode_unified_returns)
    mean_env = np.mean(episode_env_returns)
    std_env = np.std(episode_env_returns)

    print(f"\n{name} Results over {n_episodes} episodes:")
    print(f"  Unified reward mean ± std : {mean_unified:.2f} ± {std_unified:.2f}")
    print(f"  Env reward mean ± std     : {mean_env:.2f} ± {std_env:.2f}\n")

    return mean_unified, mean_env, trajectories


# =========================================================
# Run Evaluation
# =========================================================
print("\n=== Evaluating Domain-Randomized PPO on MuJoCo (Zero-Shot) ===")
mjc_unified, mjc_envr, mjc_trajs = evaluate_env(mjc_env, n_episodes=10, name="MuJoCo (DR-trained zero-shot)")


# =========================================================
# Plot Reward Trajectories
# =========================================================
def plot_trajectories(trajs, title="MuJoCo DR-Trained Zero-Shot Trajectories"):
    """
    Plot per-step unified rewards averaged across episodes.
    """
    max_len = max(len(t) for t in trajs)
    def pad_and_stack(trajs):
        padded = [t[:max_len] if len(t) >= max_len else np.pad(t, (0, max_len - len(t))) for t in trajs]
        return np.array(padded)

    traj_mat = pad_and_stack(trajs)
    mean = traj_mat.mean(axis=0)
    std = traj_mat.std(axis=0)

    plt.figure(figsize=(8, 5))
    x = np.arange(max_len)
    plt.plot(x, mean, label="MuJoCo (DR-trained zero-shot)", color="orange")
    plt.fill_between(x, mean - std, mean + std, color="orange", alpha=0.2)
    plt.title(title)
    plt.xlabel("Timestep")
    plt.ylabel("Unified Reward (mean ± std)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("dr_mujoco_zero_shot.png", dpi=200)
    plt.show()
    print("📈 Saved plot: dr_mujoco_zero_shot.png")

plot_trajectories(mjc_trajs)
