import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

# -----------------------------
# Paths and Parameters
# -----------------------------
PYBULLET_MODEL_PATH = "../ppo_walker_modern"  # Trained PyBullet model
MUJOCO_ENV_ID = "Walker2d-v5"              # Modern MuJoCo equivalent
MAX_EPISODES = 50                          # Number of test episodes
USE_RANDOMIZATION = False                  # Optional domain randomization

# -----------------------------
# Optional: Domain Randomization Wrapper
# -----------------------------
class RandomizedWalker2d(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if USE_RANDOMIZATION:
            # Randomize mass, damping, friction, gravity
            scale_mass = np.random.uniform(0.8, 1.2)
            scale_friction = np.random.uniform(0.8, 1.2)
            scale_damping = np.random.uniform(0.8, 1.2)
            self.env.model.body_mass[:] *= scale_mass
            self.env.model.geom_friction[:, 0] *= scale_friction
            self.env.model.dof_damping[:] *= scale_damping
            self.env.model.opt.gravity[:] = [0, 0, -9.81 * np.random.uniform(0.9, 1.1)]
        return obs, info

# -----------------------------
# Create MuJoCo environment
# -----------------------------
env = gym.make(MUJOCO_ENV_ID, render_mode="human")
env = RandomizedWalker2d(env)

# -----------------------------
# Load PyBullet-trained model
# -----------------------------
if not os.path.exists(PYBULLET_MODEL_PATH + ".zip"):
    raise FileNotFoundError(
        f"Model not found: {PYBULLET_MODEL_PATH}.zip\n"
        "Train it in PyBullet first or place it in the working directory."
    )

model = PPO.load(PYBULLET_MODEL_PATH)
print(f"✅ Loaded model from {PYBULLET_MODEL_PATH}.zip")

# -----------------------------
# Observation Adapter (if needed)
# -----------------------------
# MuJoCo and PyBullet Walker2D usually have slightly different obs dims (e.g., 22 vs 17).
# We'll pad or crop automatically for compatibility.

def adapt_observation(obs, target_dim):
    if obs.shape[0] > target_dim:
        return obs[:target_dim]
    elif obs.shape[0] < target_dim:
        return np.concatenate([obs, np.zeros(target_dim - obs.shape[0])])
    else:
        return obs

# -----------------------------
# Test Model in MuJoCo
# -----------------------------
obs, info = env.reset()
episodes = 0

# Get PyBullet model's expected observation dimension
policy_obs_dim = model.observation_space.shape[0] if hasattr(model, "observation_space") else len(obs)

while episodes < MAX_EPISODES:
    adapted_obs = adapt_observation(obs, policy_obs_dim)
    action, _ = model.predict(adapted_obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        episodes += 1
        print(f"Episode {episodes} finished.")
        obs, info = env.reset()

env.close()
print("🏁 Transfer test completed successfully.")
