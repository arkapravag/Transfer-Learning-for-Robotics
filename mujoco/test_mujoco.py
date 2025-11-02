import gymnasium as gym

env = gym.make("Walker2d-v5", render_mode="human")  # v5 is the new one
obs, info = env.reset()

for step in range(500):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        obs, info = env.reset()

env.close()
print("✅ MuJoCo test completed successfully.")
