import os
import gymnasium as gym
import pybullet_envs_gymnasium
from stable_baselines3 import PPO

# -----------------------------
# Parameters
# -----------------------------
ENV_ID = "Walker2DBulletEnv-v0"
MODEL_PATH = "ppo_walker_modern"
TOTAL_TIMESTEPS = 1_000_000   # Long training (~1000+ episodes)
MAX_EPISODES = 1000           # Run at least 1000 episodes after training

# -----------------------------
# Create environment with live display
# -----------------------------
# Use "human" so the simulator window opens and runs in real-time
env = gym.make(ENV_ID, render_mode="human")

# -----------------------------
# Train PPO if model doesn't exist
# -----------------------------
if os.path.exists(MODEL_PATH + ".zip"):
    model = PPO.load(MODEL_PATH, env=env)
    print("Loaded existing model.")
else:
    print("No model found. Starting training...")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(MODEL_PATH)
    print(f"Training complete. Model saved as {MODEL_PATH}.zip")

# -----------------------------
# Run the trained agent with live rendering
# -----------------------------
obs, info = env.reset()
episodes = 0

while episodes < MAX_EPISODES:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    # Rendering happens automatically in "human" mode
    if terminated or truncated:
        episodes += 1
        print(f"Episode {episodes} finished.")
        obs, info = env.reset()

# -----------------------------
# Cleanup
# -----------------------------
env.close()
print("Done.")

