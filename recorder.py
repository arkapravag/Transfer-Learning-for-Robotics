import pybullet as p
import pybullet_envs
import gym
from stable_baselines3 import PPO
import cv2
import numpy as np

# Connect to GUI
p.connect(p.GUI)

env = gym.make("Walker2DBulletEnv-v0")
model = PPO.load("ppo_pybullet_walker")

obs = env.reset()
frames = []

for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    
    # Render a frame (PyBullet GUI)
    frame = np.array(p.getCameraImage(640, 480)[2])[:, :, :3]
    frames.append(frame)
    
    if done:
        obs = env.reset()

# Save video
out = cv2.VideoWriter('walker_video.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (640,480))
for f in frames:
    out.write(f)
out.release()
p.disconnect()

