import os
import gym
import pybullet_envs
from stable_baselines3 import PPO
import numpy as np
import cv2

# -----------------------------
# Parameters
# -----------------------------
ENV_ID = "HopperBulletEnv-v0"
MODEL_PATH = "ppo_hopper"
VIDEO_PATH = "hopper_video.avi"
TOTAL_TIMESTEPS = 300_000  # Hopper can benefit from a bit more training
VIDEO_FRAMES = 1000
CAM_WIDTH = 640
CAM_HEIGHT = 480
FPS = 30

# -----------------------------
# Create environment
# -----------------------------
env = gym.make(ENV_ID, render=True)

# Manually set the render resolution
try:
    env.unwrapped._render_width = CAM_WIDTH
    env.unwrapped._render_height = CAM_HEIGHT
except AttributeError:
    print("Warning: Could not set _render_width/_render_height. Video may have default resolution.")

# -----------------------------
# Train PPO if model doesn't exist
# -----------------------------
if os.path.exists(MODEL_PATH + ".zip"):
    model = PPO.load(MODEL_PATH)
    print("Loaded existing model.")
else:
    print("No model found. Starting training...")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(MODEL_PATH)
    print(f"Training complete. Model saved as {MODEL_PATH}.zip")

# -----------------------------
# Run the trained agent and record frames
# -----------------------------
obs = env.reset()
frames = []
print("Recording video...")

for _ in range(VIDEO_FRAMES):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    frame = env.render(mode="rgb_array")
    frames.append(frame)
    if done:
        obs = env.reset()

# -----------------------------
# Save video
# -----------------------------
print("Converting frames and saving video...")
bgr_frames = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames]

if len(bgr_frames) > 0:
    frame_height, frame_width, _ = bgr_frames[0].shape
    out = cv2.VideoWriter(VIDEO_PATH, cv2.VideoWriter_fourcc(*'XVID'), FPS, (frame_width, frame_height))
    for f in bgr_frames:
        out.write(f)
    out.release()
    print(f"Video saved as {VIDEO_PATH}")
else:
    print("Error: No frames were recorded.")

# -----------------------------
# Cleanup
# -----------------------------
env.close()
print("Done.")

