import os
import gym
import pybullet_envs
# We still don't need to import pybullet (p) manually
from stable_baselines3 import PPO
import numpy as np
import cv2

# -----------------------------
# Parameters
# -----------------------------
ENV_ID = "Walker2DBulletEnv-v0"
MODEL_PATH = "ppo_walker"
VIDEO_PATH = "walker_video.avi"
TOTAL_TIMESTEPS = 200_000
VIDEO_FRAMES = 1000
CAM_WIDTH = 640
CAM_HEIGHT = 480
FPS = 30

# -----------------------------
# Create environment
# -----------------------------
# render=True is the old way, but it's what pops up the window.
# The deprecation warning suggests render_mode='human', but we'll stick
# with this for now as it's what you had.
env = gym.make(ENV_ID, render=True) 

# --- ADDED: Manually set the render resolution ---
# We access the "unwrapped" base environment and set these
# attributes to match your desired camera size.
try:
    env.unwrapped._render_width = CAM_WIDTH
    env.unwrapped._render_height = CAM_HEIGHT
except AttributeError:
    print("Warning: Could not set _render_width/_render_height. Video may have default resolution.")
# --------------------------------------------------

# -----------------------------
# Train PPO if model doesn't exist
# -----------------------------
if os.path.exists(MODEL_PATH + ".zip"):
    model = PPO.load(MODEL_PATH)
    print("Loaded existing model.")
else:
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(MODEL_PATH)
    print("Training complete. Model saved.")

# -----------------------------
# Run the trained agent and record frames
# -----------------------------
obs = env.reset()
frames = []

for _ in range(VIDEO_FRAMES):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    # --- MODIFIED: Call render() with *only* the mode ---
    # It will now use the _render_width/height we just set.
    frame = env.render(mode="rgb_array")
    # ----------------------------------------------------
    
    frames.append(frame)
    
    if done:
        obs = env.reset()

# -----------------------------
# Save video
# -----------------------------
# Convert RGB (from env.render) to BGR (for OpenCV)
print("Converting frames to BGR for video encoding...")
bgr_frames = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames]

# --- MODIFIED: Ensure VideoWriter size matches ---
# We use the actual size of the *first frame* to be safe,
# in case the attribute setting failed.
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
