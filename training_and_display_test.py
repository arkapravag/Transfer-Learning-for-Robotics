import os
# The main fix: import the maintained 'gymnasium' library instead of 'gym'
import gymnasium as gym
import pybullet_envs
from stable_baselines3 import PPO
import numpy as np
import cv2

# -----------------------------
# Parameters
# -----------------------------
ENV_ID = "Walker2DBulletEnv-v0"
MODEL_PATH = "ppo_walker_modern"
VIDEO_PATH = "walker_video_modern.avi"
TOTAL_TIMESTEPS = 200_000
EPISODES_TO_RECORD = 1000 # Record for 1000 full episodes
CAM_WIDTH = 640
CAM_HEIGHT = 480
FPS = 30
RECORD_VIDEO = False # Set to True to record a video after training

# -----------------------------
# Create environment with modern API
# -----------------------------
if RECORD_VIDEO:
    # 'rgb_array' is used for capturing frames for video.
    env = gym.make(ENV_ID, render_mode="rgb_array")
    # Note: The manual setting of _render_width/_height may not be necessary
    # with Gymnasium, but we leave it for compatibility with pybullet_envs.
    try:
        env.unwrapped._render_width = CAM_WIDTH
        env.unwrapped._render_height = CAM_HEIGHT
    except AttributeError:
        print("Warning: Could not set _render_width/_render_height.")
else:
    # If not recording, we don't need a render mode for training.
    # For live viewing, you could use render_mode="human".
    env = gym.make(ENV_ID)


# -----------------------------
# Train PPO if model doesn't exist
# -----------------------------
if os.path.exists(MODEL_PATH + ".zip"):
    # Pass the env to the loaded model
    model = PPO.load(MODEL_PATH, env=env)
    print("Loaded existing model.")
else:
    print("No model found. Starting training...")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(MODEL_PATH)
    print(f"Training complete. Model saved as {MODEL_PATH}.zip")

# -----------------------------
# Run the trained agent and record episodes (if enabled)
# -----------------------------
if RECORD_VIDEO:
    obs, info = env.reset()
    frames = []
    print(f"Recording video for {EPISODES_TO_RECORD} episodes...")

    episodes_recorded = 0
    while episodes_recorded < EPISODES_TO_RECORD:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        frame = env.render()
        frames.append(frame)
        
        # An episode ends if the environment signals terminated or truncated
        if terminated or truncated:
            episodes_recorded += 1
            print(f"Episode {episodes_recorded} finished.")
            obs, info = env.reset()

    # -----------------------------
    # Save video
    # -----------------------------
    print("Converting frames and saving video...")
    bgr_frames = [cv2.cvtColor(f, cv2.COLOR_RGB_BGR) for f in frames]

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


