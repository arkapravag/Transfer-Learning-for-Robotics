# train_pybullet.py
import gym
import pybullet_envs  # registers PyBullet environments
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Choose environment
env_id = "Walker2DBulletEnv-v0"  # or "HopperBulletEnv-v0"
env = gym.make(env_id)

# Create PPO model
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    verbose=1,
)

# Train policy
model.learn(total_timesteps=2_000_000)

# Save model
model.save("ppo_pybullet_walker")

# Evaluate in PyBullet
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"PyBullet {env_id} → mean reward {mean_reward} ± {std_reward}")
